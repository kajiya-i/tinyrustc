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
//! Every read of an input is funnelled through an accessor that records a
//! [`DepKey`] into the frame of the query currently executing. Those keys are
//! stored alongside the memoized value, and a memo is considered valid when
//! none of its recorded dependencies has changed since the memo was verified.
//! Reads therefore may not bypass the accessors: touching a field directly
//! silently drops an edge from the dependency graph and produces stale results.
//!
//! # Current limitations
//!
//! - Validity is decided by comparing revisions, never values. Rewriting an
//!   input with an identical value still invalidates everything derived from
//!   it. Backdating fixes this.
//! - Only inputs can be dependencies. Once one derived query calls another,
//!   [`DepKey`] must grow a variant per query and validation must become
//!   recursive.

use std::collections::HashMap;

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

/// Identifies something a query is allowed to read.
///
/// One variant per input kind today; derived queries will need variants of
/// their own once they can depend on each other.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum DepKey {
    SourceText(FileId),
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

/// A memoized query result, the revision at which it was last known good, and
/// everything it read while being computed.
struct Memo<V> {
    value: V,
    verified_at: Revision,
    deps: Vec<DepKey>,
}

/// Accumulates the dependencies of one in-flight query.
///
/// Frames form a stack so that a query calling another query does not steal its
/// callee's reads.
struct QueryFrame {
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

    /// Returns the revision at which `dep` last changed.
    fn changed_at(&self, dep: DepKey) -> Revision {
        match dep {
            DepKey::SourceText(file) => self
                .source_text_changed_at
                .get(&file)
                .copied()
                .unwrap_or(Revision::START),
        }
    }

    /// Whether every dependency in `deps` predates `verified_at`.
    ///
    /// An empty dependency set is trivially unchanged: a query that read
    /// nothing can never go stale.
    fn deps_unchanged(&self, deps: &[DepKey], verified_at: Revision) -> bool {
        deps.iter().all(|&dep| self.changed_at(dep) <= verified_at)
    }

    /// Counts whitespace-separated tokens in `file`.
    ///
    /// A placeholder computation. What matters is the surrounding pattern:
    /// consult the memo table, return early if its dependencies still hold,
    /// otherwise run the body under a fresh frame and store what it read.
    pub fn token_count(&mut self, file: FileId) -> usize {
        let key = format!("token_count({})", file.0);

        // FIXME: validity is computed in a separate pass over the memo table so
        // that the borrow ends before the table is mutated, costing a second
        // lookup on the hit path.
        let valid = match self.token_count_memos.get(&file) {
            Some(memo) => self.deps_unchanged(&memo.deps, memo.verified_at),
            None => false,
        };

        if valid {
            let value = self.token_count_memos[&file].value;
            self.events.push(Event::Reused(key));
            return value;
        }

        self.events.push(Event::Executed(key));
        self.stack.push(QueryFrame { deps: Vec::new() });
        let value = self.token_count_impl(file);
        let frame = self.stack.pop().expect("query stack underflow");

        let verified_at = self.current;
        self.token_count_memos.insert(
            file,
            Memo {
                value,
                verified_at,
                deps: frame.deps,
            },
        );
        value
    }

    /// The body of [`Db::token_count`], separated so that the memoization
    /// wrapper stays free of domain logic.
    fn token_count_impl(&mut self, file: FileId) -> usize {
        self.source_text(file).split_whitespace().count()
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
