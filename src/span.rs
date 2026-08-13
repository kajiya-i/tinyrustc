//! Byte positions and source ranges.
//!
//! # Divergence from `rustc`
//!
//! In `rustc`, a `BytePos` is an offset into a single address space spanning
//! every source file at once: the `SourceMap` hands each file a `start_pos`, and
//! a `Span` can therefore be resolved to a file without any surrounding context.
//! That is what lets `Span` stay eight bytes wide and carry no file identifier.
//!
//! The price is that inserting one byte into an early file shifts every position
//! in every later file. `rustc` pays for that with stable file identifiers and
//! relative offsets in its incremental fingerprints.
//!
//! Here a `BytePos` is an offset into one file. Every query is keyed by a
//! `FileId`, so the file is always known from context and the global space buys
//! nothing. Positions in one file are then unaffected by edits to any other.

use std::ops::Add;

/// A byte offset into one source file.
///
/// `u32` follows `rustc`'s `BytePos`, capping a single file at four gibibytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BytePos(pub u32);

impl BytePos {
    /// The start of a file.
    pub const ZERO: BytePos = BytePos(0);
}

impl Add<u32> for BytePos {
    type Output = BytePos;

    fn add(self, rhs: u32) -> BytePos {
        BytePos(self.0 + rhs)
    }
}

/// A half-open byte range `[lo, hi)` within one source file.
///
/// Attached to everything the compiler can complain about. Diagnostics are only
/// as good as the spans reaching them, so nearly every syntactic and semantic
/// item carries one.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Span {
    lo: BytePos,
    hi: BytePos,
}

impl Span {
    /// The span of something the compiler generated, which corresponds to no
    /// source text.
    ///
    /// `rustc` calls this `DUMMY_SP`. Desugaring produces nodes that have to
    /// carry a span but have nothing to point at; pointing them at the start of
    /// the file is the least confusing available lie.
    pub const DUMMY: Span = Span {
        lo: BytePos::ZERO,
        hi: BytePos::ZERO,
    };

    /// # Panics
    ///
    /// Panics in debug builds if `hi` precedes `lo`. An inverted span is always
    /// a bug in whoever computed it, and it surfaces far from its cause if it is
    /// allowed to propagate.
    pub fn new(lo: BytePos, hi: BytePos) -> Span {
        debug_assert!(lo <= hi, "inverted span: {lo:?}..{hi:?}");
        Span { lo, hi }
    }

    pub fn lo(self) -> BytePos {
        self.lo
    }

    pub fn hi(self) -> BytePos {
        self.hi
    }

    pub fn len(self) -> u32 {
        self.hi.0 - self.lo.0
    }

    pub fn is_empty(self) -> bool {
        self.lo == self.hi
    }

    /// The smallest span covering both `self` and `other`.
    ///
    /// The workhorse of the parser: a node's span is the union of the spans of
    /// its first and last tokens. Argument order does not matter.
    pub fn to(self, other: Span) -> Span {
        Span {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// The text this span covers within `text`.
    ///
    /// `rustc` reaches this through `SourceMap::span_to_snippet`, because there
    /// the file has to be located first.
    ///
    /// # Panics
    ///
    /// Panics if the span is out of bounds for `text`, or if either end falls
    /// inside a multi-byte character.
    pub fn slice(self, text: &str) -> &str {
        &text[self.lo.0 as usize..self.hi.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(lo: u32, hi: u32) -> Span {
        Span::new(BytePos(lo), BytePos(hi))
    }

    #[test]
    fn a_span_reports_its_length() {
        assert_eq!(span(3, 7).len(), 4);
        assert_eq!(span(3, 3).len(), 0);
    }

    #[test]
    fn only_a_zero_length_span_is_empty() {
        assert!(span(4, 4).is_empty());
        assert!(!span(4, 5).is_empty());
        assert!(Span::DUMMY.is_empty());
    }

    #[test]
    fn union_covers_both_operands() {
        assert_eq!(span(2, 4).to(span(8, 9)), span(2, 9));
        // Order must not matter, because the parser does not always visit
        // tokens left to right.
        assert_eq!(span(8, 9).to(span(2, 4)), span(2, 9));
    }

    #[test]
    fn union_of_nested_spans_is_the_outer_one() {
        assert_eq!(span(0, 10).to(span(3, 4)), span(0, 10));
    }

    #[test]
    fn slicing_recovers_the_source_text() {
        let text = "fn main() {}";
        assert_eq!(span(0, 2).slice(text), "fn");
        assert_eq!(span(3, 7).slice(text), "main");
        assert_eq!(span(0, 0).slice(text), "");
    }

    #[test]
    fn byte_positions_advance_by_token_length() {
        // How the layer above turns raw token lengths into absolute positions.
        let mut pos = BytePos::ZERO;
        for len in [2u32, 1, 4, 2] {
            pos = pos + len;
        }
        assert_eq!(pos, BytePos(9));
    }
}
