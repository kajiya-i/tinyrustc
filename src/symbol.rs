//! Interned strings.
//!
//! Identifiers are compared constantly - once per name lookup, per method
//! resolution, per field access - so they are replaced by indices into a table
//! and compared as integers. `rustc`'s `Symbol` is the same device.
//!
//! Keywords occupy the lowest indices, assigned in [`Interner::new`] before any
//! identifier from a source file can be interned. Recognising a keyword is then
//! also an integer comparison, and `rustc` generates that numbering with its
//! `symbols!` macro rather than writing it out.
//!
//! A [`Symbol`] deliberately implements neither `Ord` nor `Display`. Ordering
//! would be by interning order, which depends on the order files happen to be
//! read - sorting by it would make output depend on scheduling, exactly the kind
//! of instability that breaks incremental compilation. Recovering the text needs
//! the [`Interner`] that issued the symbol, so it cannot be a method on the
//! symbol itself.

use std::collections::HashMap;

/// An interned string, identified by its position in an [`Interner`].
///
/// Only an [`Interner`] can produce one, so a symbol is always valid for the
/// interner that issued it - and meaningless for any other.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Symbol(u32);

/// The keywords of this language.
///
/// The indices must match the order of `KEYWORDS`, which the tests below check
/// exhaustively.
pub mod kw {
    use super::Symbol;

    pub const FN: Symbol = Symbol(0);
    pub const LET: Symbol = Symbol(1);
    pub const MUT: Symbol = Symbol(2);
    pub const IF: Symbol = Symbol(3);
    pub const ELSE: Symbol = Symbol(4);
    pub const WHILE: Symbol = Symbol(5);
    pub const RETURN: Symbol = Symbol(6);
    pub const STRUCT: Symbol = Symbol(7);
    pub const TRUE: Symbol = Symbol(8);
    pub const FALSE: Symbol = Symbol(9);
}

/// Keyword spellings, in the order the [`kw`] constants number them.
const KEYWORDS: [&str; 10] = [
    "fn", "let", "mut", "if", "else", "while", "return", "struct", "true", "false",
];

impl Symbol {
    /// Whether this symbol is one of the language's keywords.
    ///
    /// A range check, because keywords are interned first and nothing else can
    /// land below them.
    pub fn is_keyword(self) -> bool {
        (self.0 as usize) < KEYWORDS.len()
    }
}

/// The table mapping strings to [`Symbol`]s and back.
pub struct Interner {
    /// Indexed by `Symbol`, so `strings[symbol.0]` is its text.
    strings: Vec<String>,
    /// FIXME: every interned string is stored twice, once here as a key and once
    /// in `strings`. `rustc` allocates the text in an arena and keeps `&str`
    /// views of it in both places.
    lookup: HashMap<String, Symbol>,
}

impl Interner {
    /// Builds an interner with the keywords already assigned.
    pub fn new() -> Interner {
        let mut interner = Interner {
            strings: Vec::new(),
            lookup: HashMap::new(),
        };
        for keyword in KEYWORDS {
            interner.intern(keyword);
        }
        interner
    }

    /// Returns the symbol for `text`, assigning a fresh one if it is new.
    pub fn intern(&mut self, text: &str) -> Symbol {
        if let Some(&symbol) = self.lookup.get(text) {
            return symbol;
        }
        let symbol = Symbol(self.strings.len() as u32);
        self.strings.push(text.to_string());
        self.lookup.insert(text.to_string(), symbol);
        symbol
    }

    /// The text `symbol` stands for.
    ///
    /// # Panics
    ///
    /// Panics if `symbol` came from a different interner.
    pub fn get(&self, symbol: Symbol) -> &str {
        &self.strings[symbol.0 as usize]
    }
}

impl Default for Interner {
    fn default() -> Interner {
        Interner::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `kw` constants, in the order `KEYWORDS` spells them.
    const KEYWORD_SYMBOLS: [Symbol; 10] = [
        kw::FN,
        kw::LET,
        kw::MUT,
        kw::IF,
        kw::ELSE,
        kw::WHILE,
        kw::RETURN,
        kw::STRUCT,
        kw::TRUE,
        kw::FALSE,
    ];

    /// The one thing that can silently break: a constant naming the wrong
    /// index. Nothing else in the crate would notice.
    #[test]
    fn keyword_constants_match_their_spellings() {
        let interner = Interner::new();
        for (symbol, spelling) in KEYWORD_SYMBOLS.into_iter().zip(KEYWORDS) {
            assert_eq!(interner.get(symbol), spelling);
        }
    }

    #[test]
    fn interning_the_same_text_twice_yields_the_same_symbol() {
        let mut interner = Interner::new();
        let first = interner.intern("main");
        let second = interner.intern("main");
        assert_eq!(first, second);
    }

    #[test]
    fn distinct_texts_yield_distinct_symbols() {
        let mut interner = Interner::new();
        assert_ne!(interner.intern("main"), interner.intern("other"));
    }

    #[test]
    fn interning_a_keyword_returns_its_constant() {
        let mut interner = Interner::new();
        assert_eq!(interner.intern("fn"), kw::FN);
        assert_eq!(interner.intern("false"), kw::FALSE);
    }

    #[test]
    fn only_keywords_are_keywords() {
        let mut interner = Interner::new();
        for symbol in KEYWORD_SYMBOLS {
            assert!(symbol.is_keyword());
        }
        assert!(!interner.intern("main").is_keyword());
        // A word that merely contains a keyword is not one.
        assert!(!interner.intern("iffy").is_keyword());
        assert!(!interner.intern("Fn").is_keyword());
    }

    #[test]
    fn symbols_round_trip_through_the_interner() {
        let mut interner = Interner::new();
        let symbol = interner.intern("\u{3042}\u{3044}");
        assert_eq!(interner.get(symbol), "\u{3042}\u{3044}");
    }
}
