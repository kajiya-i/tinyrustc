//! Tests that the query system reuses memoized values when it should, and
//! recomputes them when it must.

use std::collections::BTreeSet;

use tinyrustc::db::{Db, Event, FileId};
use tinyrustc::token::TokenKind;

/// Two whitespace-separated tokens: trivial, weight 0.
const TWO: &str = "a b";

/// Four whitespace-separated tokens: not trivial, weight 4.
const FOUR: &str = "a b c d";

/// Five whitespace-separated tokens: not trivial, weight 5.
const FIVE: &str = "a b c d e";

fn executed(name: &str) -> Event {
    Event::Executed(name.to_string())
}

fn reused(name: &str) -> Event {
    Event::Reused(name.to_string())
}

/// The set of queries whose bodies ran.
///
/// Multi-level graphs are validated and computed in an order that is an
/// implementation detail, so asserting on the exact event sequence would pin
/// down more than these tests mean to. What matters is which bodies had to run.
fn executed_queries(events: &[Event]) -> BTreeSet<&str> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Executed(name) => Some(name.as_str()),
            Event::Reused(_) => None,
        })
        .collect()
}

fn query_set<const N: usize>(names: [&str; N]) -> BTreeSet<&str> {
    names.into_iter().collect()
}

#[test]
fn first_call_executes() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    assert_eq!(db.token_count(f), 4);
    assert_eq!(db.take_events(), vec![executed("token_count(0)")]);
}

#[test]
fn second_call_reuses_memo() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.token_count(f);
    db.take_events();

    // Nothing was written, so the memo must still be valid.
    assert_eq!(db.token_count(f), 4);
    assert_eq!(db.take_events(), vec![reused("token_count(0)")]);
}

#[test]
fn input_change_forces_reexecution() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.token_count(f);
    db.take_events();

    db.set_source_text(f, FIVE.to_string());

    assert_eq!(db.token_count(f), 5);
    assert_eq!(db.take_events(), vec![executed("token_count(0)")]);
}

/// The point of dependency tracking: a query is only invalidated by the inputs
/// it actually read.
#[test]
fn unrelated_input_is_not_a_dependency() {
    let mut db = Db::new();
    let f0 = FileId(0);
    let f1 = FileId(1);
    db.set_source_text(f0, FOUR.to_string());
    db.set_source_text(f1, FIVE.to_string());

    db.token_count(f0);
    db.take_events();

    // token_count(f0) never read f1, so writing f1 must not disturb it.
    db.set_source_text(f1, FOUR.to_string());

    assert_eq!(db.token_count(f0), 4);
    assert_eq!(db.take_events(), vec![reused("token_count(0)")]);
}

// ---- Derived queries reading derived queries ----

#[test]
fn cold_graph_executes_every_query() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    assert_eq!(db.weight(f), 4);

    let events = db.take_events();
    assert_eq!(
        executed_queries(&events),
        query_set(["weight(0)", "token_count(0)", "is_trivial(0)"]),
    );
}

/// A memo already verified in this revision is accepted without walking its
/// dependencies, so a repeated call touches no query body at all.
#[test]
fn warm_graph_executes_nothing() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.weight(f);
    db.take_events();

    assert_eq!(db.weight(f), 4);

    let events = db.take_events();
    assert!(executed_queries(&events).is_empty());
    assert_eq!(events, vec![reused("weight(0)")]);
}

/// Invalidation propagates transitively: `weight` reads `is_trivial`, which
/// reads `token_count`, which reads the input.
#[test]
fn input_change_propagates_through_the_graph() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.weight(f);
    db.take_events();

    db.set_source_text(f, TWO.to_string());

    // Two tokens is trivial, so the weight collapses to zero.
    assert_eq!(db.weight(f), 0);

    let events = db.take_events();
    assert_eq!(
        executed_queries(&events),
        query_set(["weight(0)", "token_count(0)", "is_trivial(0)"]),
    );
}

#[test]
fn unrelated_file_leaves_the_graph_warm() {
    let mut db = Db::new();
    let f0 = FileId(0);
    let f1 = FileId(1);
    db.set_source_text(f0, FOUR.to_string());
    db.set_source_text(f1, FOUR.to_string());

    db.weight(f0);
    db.take_events();

    db.set_source_text(f1, TWO.to_string());

    assert_eq!(db.weight(f0), 4);

    let events = db.take_events();
    assert!(executed_queries(&events).is_empty());
}

/// Backdating cannot spare the query that sits directly on the changed input.
///
/// The input's revision moved, and nothing short of running the body can
/// establish whether the text actually differs, so `token_count` re-executes.
/// What backdating buys is that the change stops here.
#[test]
fn identical_rewrite_still_reruns_the_first_query() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.token_count(f);
    db.take_events();

    // Same bytes as before.
    db.set_source_text(f, FOUR.to_string());

    db.token_count(f);
    assert_eq!(db.take_events(), vec![executed("token_count(0)")]);
}

/// Backdating stops a change from propagating; it does not stop a query whose
/// dependency moved from running.
///
/// Four tokens to five genuinely changes `token_count`, so `is_trivial` has to
/// run to find out whether its own answer moved. It recomputes `false` and
/// backdates, but `weight` reads `token_count` directly and re-runs regardless,
/// which is why the backdating is not observable from here.
#[test]
fn backdating_does_not_spare_the_query_itself() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.weight(f);
    db.take_events();

    db.set_source_text(f, FIVE.to_string());

    assert_eq!(db.weight(f), 5);

    let events = db.take_events();
    assert!(executed_queries(&events).contains("is_trivial(0)"));
}

/// The payoff. A no-op write invalidates `token_count`, which recomputes the same count and backdates, so validation of `is_trivial` and `weight`
/// succeeds without either body running - two derived levels stay green.
#[test]
fn identical_rewrite_stops_at_the_first_query() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.weight(f);
    db.take_events();

    // Same bytes as before.
    db.set_source_text(f, FOUR.to_string());

    assert_eq!(db.weight(f), 4);

    let events = db.take_events();
    assert_eq!(executed_queries(&events), query_set(["token_count(0)"]));
}

// ---- The interner is not part of the query graph ----

/// Interning is idempotent within a database, which is what lets symbol
/// equality stand in for string equality everywhere else.
#[test]
fn interning_is_stable_within_a_database() {
    let db = Db::new();

    let first = db.intern("main");
    let second = db.intern("main");
    assert_eq!(first, second);
    assert_eq!(&*db.symbol_text(first), "main");
}

/// Interning must not attach a dependency edge to whichever query is running,
/// or adding one unrelated identifier would invalidate unrelated memos.
#[test]
fn interning_does_not_disturb_memos() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, FOUR.to_string());

    db.weight(f);
    db.take_events();

    db.intern("something_new");

    assert_eq!(db.weight(f), 4);

    let events = db.take_events();
    assert!(executed_queries(&events).is_empty());
}

// ---- Lexing through the query system ----

/// A short program used for the lexing tests. Seven tokens including `Eof`.
const PROGRAM: &str = "fn main() {}";

#[test]
fn lexing_is_memoized() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, PROGRAM.to_string());

    let first = db.lexed(f);
    assert_eq!(db.take_events(), vec![executed("lexed(0)")]);

    let second = db.lexed(f);
    assert_eq!(db.take_events(), vec![reused("lexed(0)")]);
    assert_eq!(first, second);
}

/// Names interned while lexing land in the database's interner, so a symbol
/// carried by a token resolves through it.
#[test]
fn lexing_interns_into_the_database() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, "answer".to_string());

    let lexed = db.lexed(f);
    match lexed.tokens[0].kind {
        TokenKind::Ident(symbol) => assert_eq!(&*db.symbol_text(symbol), "answer"),
        other => panic!("expected an identifier, got {other:?}"),
    }
}

/// Errors travel with the tokens rather than stopping lexing, because a parser
/// needs the rest of the input to recover.
#[test]
fn errors_arrive_alongside_the_tokens() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, "a # b".to_string());

    let lexed = db.lexed(f);
    assert_eq!(lexed.errors.len(), 1);
    // Ident, Ident, Eof - the unknown character is dropped, not fatal.
    assert_eq!(lexed.tokens.len(), 3);
}

/// Pins the cost of putting absolute positions in a query result: one leading
/// space shifts every span, so the value differs and backdating has nothing to
/// backdate.
#[test]
fn a_leading_space_changes_every_span() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, PROGRAM.to_string());
    let before = db.lexed(f);

    db.set_source_text(f, format!(" {PROGRAM}"));
    let after = db.lexed(f);

    assert_eq!(before.tokens.len(), after.tokens.len());
    assert_ne!(before, after);
}

/// Rewriting a file with identical bytes re-lexes it, but the value is equal so
/// the change revision is backdated.
///
/// Not observable from outside yet, because nothing reads `lexed`. The parser
/// will make it visible.
#[test]
fn identical_rewrite_relexes_but_backdates() {
    let mut db = Db::new();
    let f = FileId(0);
    db.set_source_text(f, PROGRAM.to_string());
    let before = db.lexed(f);
    db.take_events();

    db.set_source_text(f, PROGRAM.to_string());
    let after = db.lexed(f);

    assert_eq!(db.take_events(), vec![executed("lexed(0)")]);
    assert_eq!(before, after);
}
