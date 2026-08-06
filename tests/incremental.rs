//! Tests that the query system reuses memoized values when it should, and
//! recomputes them when it must.

use tinyrustc::db::{Db, Event, FileId};

/// Four whitespace-separated tokens.
const FOUR: &str = "a b c d";

/// Five whitespace-separated tokens.
const FIVE: &str = "a b c d e";

fn executed(name: &str) -> Event {
    Event::Executed(name.to_string())
}

fn reused(name: &str) -> Event {
    Event::Reused(name.to_string())
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

/// Pins down a known deficiency: rewriting an input with an identical value
/// still invalidates memos derived from it.
///
/// Validity compares revisions, never values, so a no-op write bumps the
/// input's change revision and forces recomputation. Backdating - comparing the
/// recomputed value against the previous one and restoring the older
/// verification revision when they agree - is what fixes this.
#[test]
fn identical_rewrite_still_invalidates() {
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
