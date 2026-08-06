//! Tests that the query system reuses memoized values when it should, and
//! recomputes them when it must.

use std::collections::BTreeSet;

use tinyrustc::db::{Db, Event, FileId};

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

/// Pins the deficiency that backdating exists to fix.
///
/// Going from four tokens to five genuinely changes `token_count`, but
/// `is_trivial` is false either way. Because a re-executed query always reports
/// the current revision as its `changed_at`, `is_trivial` is invalidated even
/// though its value could not have moved. Comparing the recomputed value
/// against the previous one, and keeping the older `changed_at` when they
/// agree, would let it stay green.
#[test]
fn dependent_reruns_even_when_the_value_is_unchanged() {
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
