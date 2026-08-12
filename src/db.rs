//! A minimal demand-driven query system.
//!
//! This is a hand-written, deliberately simplified version of the machinery
//! that drives `rustc` (and, in a different form, the `salsa` crate). The core
//! idea is that compilation is not a pipeline of passes but a set of pure
//! functions ("queries") whose results are memoized. Instead of running passes
//! in order, the compiler asks for the value it needs and the system computes
//! only what is required to produce it.
//!
//! Correctness of a query system is hard to eyeball, so this module exposes an
//! [`Event`] log recording whether each query was executed or served from the
//! memo table. Tests assert against that log; without it, incrementality can
//! only be guessed at.
//!
//! # Dependency tracking
//!
//! Every read is funnelled through an accessor that records a [`DepKey`] into
//! the frame of the query currently executing. Those keys are stored alongside
//! the memoized value. Reads may not bypass the accessors: touching a field
//! directly silently drops an edge from the dependency graph and produces stale
//! results.
//!
//! # Validation
//!
//! A memo carries two revisions. `changed_at` is when its value last differed;
//! `verified_at` is when it was last confirmed current. Confirming a memo means
//! bringing each of its dependencies up to date - which may execute them - and
//! checking that none reports a change newer than `verified_at`. A memo already
//! verified in this revision is accepted without walking its dependencies at
//! all, which is what keeps diamond-shaped graphs from being traversed
//! exponentially.
//!
//! # Current limitations
//!
//! - Backdating compares values with `==`, so it only helps for cheaply
//!   comparable results. Large trees will want a fingerprint instead.
//! - Cycles abort the process rather than producing a diagnostic.
//! - Every query needs its own memo table and match arm by hand. `salsa`'s
//!   attribute macros exist to generate exactly this boilerplate.

use std::collections::HashMap;
use std::fmt;

/// A monotonically increasing logical clock, bumped on every write to an input.
///
/// All invalidation decisions reduce to comparing two revisions: when a memo
/// was last verified, versus when its dependencies last changed.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Revision(u64);

impl Revision {
    /// The revision of a database that has never been written to.
    pub const START: Revision = Revision(0);

    fn next(self) -> Revision {
        Revision(self.0 + 1)
    }
}

/// Identifies a source file.
///
/// Deliberately an opaque index rather than a path or a reference: query keys
/// must be cheap to hash and stable across revisions. `DefId` will follow the
/// same shape.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FileId(pub u32);

/// Identifies something a query is allowed to read: an input or another query.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum DepKey {
    SourceText(FileId),
    TokenCount(FileId),
    IsTrivial(FileId),
    Weight(FileId),
}

impl fmt::Display for DepKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DepKey::SourceText(file) => write!(f, "source_text({})", file.0),
            DepKey::TokenCount(file) => write!(f, "token_count({})", file.0),
            DepKey::IsTrivial(file) => write!(f, "is_trivial({})", file.0),
            DepKey::Weight(file) => write!(f, "weight({})", file.0),
        }
    }
}

/// A record of how a query call was serviced.
///
/// Emitted so that tests can distinguish "returned the right answer" from
/// "returned the right answer without recomputing it". The latter is the whole
/// point of the query system and is otherwise invisible.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    /// The query body ran.
    Executed(String),
    /// A memoized value was returned.
    Reused(String),
}

/// A memoized query result, the revisions bounding its validity, and everything
/// it read while being computed.
struct Memo<V> {
    value: V,
    changed_at: Revision,
    verified_at: Revision,
    deps: Vec<DepKey>,
}

/// Accumulates the reads attributed to one in-flight query.
///
/// Frames form a stack so that a query calling another query does not steal its
/// callee's reads.
struct QueryFrame {
    /// The query being computed, or `None` for a validation frame whose
    /// recorded reads are discarded.
    key: Option<DepKey>,
    deps: Vec<DepKey>,
}

/// Storage for all inputs, memos, and the event log.
///
/// Queries are methods on this type. Reads take `&mut self` because servicing a
/// query may populate the memo table and record dependencies.
pub struct Db {
    current: Revision,

    // Inputs, paired with the revision at which each was last written.
    source_texts: HashMap<FileId, String>,
    source_text_changed_at: HashMap<FileId, Revision>,

    // Memo tables, one per derived query.
    token_count_memos: HashMap<FileId, Memo<usize>>,
    is_trivial_memos: HashMap<FileId, Memo<bool>>,
    weight_memos: HashMap<FileId, Memo<usize>>,

    stack: Vec<QueryFrame>,
    events: Vec<Event>,
}

impl Db {
    pub fn new() -> Db {
        Db {
            current: Revision::START,
            source_texts: HashMap::new(),
            source_text_changed_at: HashMap::new(),
            token_count_memos: HashMap::new(),
            is_trivial_memos: HashMap::new(),
            weight_memos: HashMap::new(),
            stack: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Sets the text of `file`, advancing the current revision.
    ///
    /// Takes `&mut self` because writing an input may invalidate values derived
    /// from it. This mirrors why `salsa`'s setters require a mutable database.
    pub fn set_source_text(&mut self, file: FileId, text: String) {
        self.current = self.current.next();
        self.source_text_changed_at.insert(file, self.current);
        self.source_texts.insert(file, text);
    }

    /// Returns the text of `file`, recording the read as a dependency.
    ///
    /// # Panics
    ///
    /// Panics if `file` has no text set. Reading an unset input is a bug in the
    /// driver, not a recoverable condition.
    fn source_text(&mut self, file: FileId) -> &str {
        self.record_dep(DepKey::SourceText(file));
        self.source_texts
            .get(&file)
            .expect("read of a FileId with no source text set")
    }

    /// Attributes a read to the query currently executing, if any.
    ///
    /// Reads issued outside a query - by the driver, say - are intentionally
    /// not recorded: they belong to no memo.
    fn record_dep(&mut self, dep: DepKey) {
        if let Some(frame) = self.stack.last_mut() {
            // Dependency sets are small, so a linear scan beats a hash set.
            if !frame.deps.contains(&dep) {
                frame.deps.push(dep);
            }
        }
    }

    /// Opens a frame for the body of `key`.
    ///
    /// # Panics
    ///
    /// Panics if `key` is already being computed further down the stack. A
    /// cyclic query graph has no least fixed point to converge on here.
    fn push_frame(&mut self, key: DepKey) {
        if self.stack.iter().any(|frame| frame.key == Some(key)) {
            panic!("query cycle detected while computing {key}");
        }
        self.stack.push(QueryFrame {
            key: Some(key),
            deps: Vec::new(),
        });
    }

    /// Closes the innermost frame and yields what it read.
    fn pop_frame(&mut self) -> Vec<DepKey> {
        self.stack.pop().expect("query stack underflow").deps
    }

    /// Brings each of `deps` up to date and reports whether all of them last
    /// changed no later than `verified_at`.
    fn deps_unchanged(&mut self, deps: &[DepKey], verified_at: Revision) -> bool {
        // Bringing a dependency up to date invokes it, and a query registers
        // itself with whatever frame is on top. Those registrations belong to
        // no memo, so they are collected into a frame that is discarded.
        // Without this, validating one query would attach spurious edges to an
        // unrelated query that happens to be executing.
        self.stack.push(QueryFrame {
            key: None,
            deps: Vec::new(),
        });

        let mut unchanged = true;
        for &dep in deps {
            if self.changed_at(dep) > verified_at {
                unchanged = false;
                break;
            }
        }

        self.stack.pop().expect("validation frame vanished");
        unchanged
    }

    /// Brings `dep` up to date and returns the revision at which its value last
    /// changed.
    ///
    /// For an input this is a stored fact. For a derived query it is not: the
    /// query has to be run, or at least revalidated, before the answer exists.
    fn changed_at(&mut self, dep: DepKey) -> Revision {
        match dep {
            DepKey::SourceText(file) => self
                .source_text_changed_at
                .get(&file)
                .copied()
                .unwrap_or(Revision::START),
            DepKey::TokenCount(file) => {
                self.token_count(file);
                self.token_count_memos
                    .get(&file)
                    .expect("memo absent after its query ran")
                    .changed_at
            }
            DepKey::IsTrivial(file) => {
                self.is_trivial(file);
                self.is_trivial_memos
                    .get(&file)
                    .expect("memo absent after its query ran")
                    .changed_at
            }
            DepKey::Weight(file) => {
                self.weight(file);
                self.weight_memos
                    .get(&file)
                    .expect("memo absent after its query ran")
                    .changed_at
            }
        }
    }

    /// Counts whitespace-separated tokens in `file`.
    pub fn token_count(&mut self, file: FileId) -> usize {
        let key = DepKey::TokenCount(file);
        self.record_dep(key);

        // FIXME: the memo is removed and reinserted so that no borrow of the
        // table is live while dependencies are validated. Interior mutability
        // would let this be a single lookup.
        let stale = match self.token_count_memos.remove(&file) {
            Some(memo) => {
                let fresh = memo.verified_at == self.current
                    || self.deps_unchanged(&memo.deps, memo.verified_at);
                if fresh {
                    let value = memo.value;
                    let verified_at = self.current;
                    self.token_count_memos.insert(
                        file,
                        Memo {
                            verified_at,
                            ..memo
                        },
                    );
                    self.events.push(Event::Reused(key.to_string()));
                    return value;
                }
                // Retained only so the recomputed value can be compared to it.
                Some(memo)
            }
            None => None,
        };

        self.events.push(Event::Executed(key.to_string()));
        // Captured before the body runs, so the memo can never claim to have
        // been verified against a revision later than the one it observed.
        let revision = self.current;
        self.push_frame(key);
        let value = self.token_count_impl(file);
        let deps = self.pop_frame();

        // Backdating. Recomputing an identical value is not a change, so the
        // older change revision stands and dependents are left green. Note that
        // this cannot spare the query itself: discovering that the value is
        // unchanged requires running the body.
        let changed_at = match &stale {
            Some(old) if old.value == value => old.changed_at,
            _ => revision,
        };

        self.token_count_memos.insert(
            file,
            Memo {
                value,
                changed_at,
                verified_at: revision,
                deps,
            },
        );
        value
    }

    fn token_count_impl(&mut self, file: FileId) -> usize {
        self.source_text(file).split_whitespace().count()
    }

    /// Whether `file` holds fewer than three tokens.
    ///
    /// Derived from another derived query rather than from an input, which is
    /// what makes validation recursive.
    pub fn is_trivial(&mut self, file: FileId) -> bool {
        let key = DepKey::IsTrivial(file);
        self.record_dep(key);

        let stale = match self.is_trivial_memos.remove(&file) {
            Some(memo) => {
                let fresh = memo.verified_at == self.current
                    || self.deps_unchanged(&memo.deps, memo.verified_at);
                if fresh {
                    let value = memo.value;
                    let verified_at = self.current;
                    self.is_trivial_memos.insert(
                        file,
                        Memo {
                            verified_at,
                            ..memo
                        },
                    );
                    self.events.push(Event::Reused(key.to_string()));
                    return value;
                }
                Some(memo)
            }
            None => None,
        };

        self.events.push(Event::Executed(key.to_string()));
        let revision = self.current;
        self.push_frame(key);
        let value = self.is_trivial_impl(file);
        let deps = self.pop_frame();

        let changed_at = match &stale {
            Some(old) if old.value == value => old.changed_at,
            _ => revision,
        };

        self.is_trivial_memos.insert(
            file,
            Memo {
                value,
                changed_at,
                verified_at: revision,
                deps,
            },
        );
        value
    }

    fn is_trivial_impl(&mut self, file: FileId) -> bool {
        self.token_count(file) < 3
    }

    /// The token count of `file`, or zero if the file is trivial.
    ///
    /// Reads two derived queries, one of which reads the other, so its
    /// dependency graph is a diamond.
    pub fn weight(&mut self, file: FileId) -> usize {
        let key = DepKey::Weight(file);
        self.record_dep(key);

        if let Some(memo) = self.weight_memos.remove(&file) {
            let fresh = memo.verified_at == self.current
                || self.deps_unchanged(&memo.deps, memo.verified_at);
            if fresh {
                let value = memo.value;
                let verified_at = self.current;
                self.weight_memos.insert(
                    file,
                    Memo {
                        verified_at,
                        ..memo
                    },
                );
                self.events.push(Event::Reused(key.to_string()));
                return value;
            }
        }

        self.events.push(Event::Executed(key.to_string()));
        let revision = self.current;
        self.push_frame(key);
        let value = self.weight_impl(file);
        let deps = self.pop_frame();
        self.weight_memos.insert(
            file,
            Memo {
                value,
                changed_at: revision,
                verified_at: revision,
                deps,
            },
        );
        value
    }

    fn weight_impl(&mut self, file: FileId) -> usize {
        // Both reads are unconditional so that the dependency set does not vary
        // with the input, which would make the tests harder to reason about.
        let count = self.token_count(file);
        if self.is_trivial(file) { 0 } else { count }
    }

    /// Drains the event log.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
    }
}
