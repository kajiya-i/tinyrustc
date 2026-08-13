//! Spanned tokens, and the second lexing layer that produces them.
//!
//! This is the counterpart to [`crate::lexer`], corresponding to
//! `rustc_ast::token` plus the cooking step in `rustc_parse::lexer`. Where the
//! raw layer answers "what and how long", this one answers "what, where, and
//! which name", and it is the first place that throws anything away.
//!
//! Three things happen here and nowhere else:
//!
//! - **Positions become absolute.** Raw token lengths are accumulated into
//!   [`Span`]s. This is sound only because the raw layer tiles its input
//!   exactly; that invariant is what makes the arithmetic here trustworthy.
//! - **Trivia is dropped.** Whitespace and comments do not survive. An
//!   IDE-oriented front end would keep them - `rust-analyzer` does, in a
//!   lossless tree - because refactoring has to reproduce them. A batch
//!   compiler does not need to.
//! - **Compound operators are assembled.** Adjacency is known here, because
//!   trivia is still visible while deciding, so `= =` stays two tokens while
//!   `==` becomes one.
//!
//! # Keywords are not token kinds
//!
//! There is no `TokenKind::Fn`. The word `fn` arrives as [`TokenKind::Ident`]
//! carrying a [`Symbol`] that happens to be a keyword, and the parser asks
//! `symbol == kw::FN`. `rustc` does the same, because keyword-ness is a property
//! of the name rather than of the token: adding a keyword then touches no lexer
//! code at all.

use crate::lexer::{RawTokenKind, tokenize};
use crate::span::{BytePos, Span};
use crate::symbol::{Interner, Symbol};

/// A token together with the source range it covers.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// What a token is, once trivia is gone and operators are assembled.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TokenKind {
    /// An identifier, which may or may not be a keyword. See the module
    /// documentation.
    Ident(Symbol),
    /// An integer literal, held as written.
    ///
    /// Deliberately not parsed into a number here. A literal too large for its
    /// type is a type error, not a lexical one, and diagnostics want the
    /// original spelling - `1_000` should not be reported as `1000`. `rustc`
    /// keeps literal text in a `Symbol` for the same reasons.
    Int(Symbol),
    /// `'a`, holding just the name: the apostrophe is syntax, not part of it.
    Lifetime(Symbol),

    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,

    Comma,
    Colon,
    Semi,
    Dot,

    Eq,
    EqEq,
    Ne,
    Bang,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    And,
    AndAnd,
    RArrow,

    /// End of input. Always the last token, with an empty span, so the parser
    /// never has to special-case running off the end.
    Eof,
}

/// Something the lexer could not make sense of.
///
/// Collected rather than reported, because this layer has no diagnostic
/// machinery yet and lexing must not stop at the first problem: a parser needs
/// the rest of the tokens to recover.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct LexError {
    pub span: Span,
    pub kind: LexErrorKind,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LexErrorKind {
    UnknownCharacter,
    UnterminatedBlockComment,
}

/// Lexes `text`, interning every name it finds.
///
/// Always returns a token list ending in [`TokenKind::Eof`], whatever errors
/// were found alongside.
pub fn lex(text: &str, interner: &mut Interner) -> (Vec<Token>, Vec<LexError>) {
    // Absolute positions come from accumulating raw token lengths. Sound only
    // because the raw layer tiles its input; see `crate::lexer`.
    let mut raw: Vec<(RawTokenKind, Span)> = Vec::new();
    let mut pos = BytePos::ZERO;
    for token in tokenize(text) {
        let end = pos + token.len;
        raw.push((token.kind, Span::new(pos, end)));
        pos = end;
    }

    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let mut index = 0;

    while index < raw.len() {
        let (kind, span) = raw[index];
        // Gluing consults the immediately following raw token, so a compound
        // operator can never be assembled across whitespace or a comment.
        let following = raw.get(index + 1).map(|&(next, _)| next);

        let (cooked, consumed) = match (kind, following) {
            (RawTokenKind::Whitespace | RawTokenKind::LineComment, _) => {
                index += 1;
                continue;
            }
            (RawTokenKind::BlockComment { terminated }, _) => {
                if !terminated {
                    errors.push(LexError {
                        span,
                        kind: LexErrorKind::UnterminatedBlockComment,
                    });
                }
                index += 1;
                continue;
            }

            (RawTokenKind::Eq, Some(RawTokenKind::Eq)) => (TokenKind::EqEq, 2),
            (RawTokenKind::Bang, Some(RawTokenKind::Eq)) => (TokenKind::Ne, 2),
            (RawTokenKind::Lt, Some(RawTokenKind::Eq)) => (TokenKind::Le, 2),
            (RawTokenKind::Gt, Some(RawTokenKind::Eq)) => (TokenKind::Ge, 2),
            (RawTokenKind::And, Some(RawTokenKind::And)) => (TokenKind::AndAnd, 2),
            (RawTokenKind::Minus, Some(RawTokenKind::Gt)) => (TokenKind::RArrow, 2),

            (RawTokenKind::Ident, _) => (TokenKind::Ident(interner.intern(span.slice(text))), 1),
            (RawTokenKind::Int, _) => (TokenKind::Int(interner.intern(span.slice(text))), 1),
            (RawTokenKind::Lifetime, _) => {
                // Skip the apostrophe the raw token includes.
                let name = &span.slice(text)[1..];
                (TokenKind::Lifetime(interner.intern(name)), 1)
            }

            (RawTokenKind::OpenParen, _) => (TokenKind::OpenParen, 1),
            (RawTokenKind::CloseParen, _) => (TokenKind::CloseParen, 1),
            (RawTokenKind::OpenBrace, _) => (TokenKind::OpenBrace, 1),
            (RawTokenKind::CloseBrace, _) => (TokenKind::CloseBrace, 1),
            (RawTokenKind::Comma, _) => (TokenKind::Comma, 1),
            (RawTokenKind::Colon, _) => (TokenKind::Colon, 1),
            (RawTokenKind::Semi, _) => (TokenKind::Semi, 1),
            (RawTokenKind::Dot, _) => (TokenKind::Dot, 1),
            (RawTokenKind::Eq, _) => (TokenKind::Eq, 1),
            (RawTokenKind::Bang, _) => (TokenKind::Bang, 1),
            (RawTokenKind::Lt, _) => (TokenKind::Lt, 1),
            (RawTokenKind::Gt, _) => (TokenKind::Gt, 1),
            (RawTokenKind::Plus, _) => (TokenKind::Plus, 1),
            (RawTokenKind::Minus, _) => (TokenKind::Minus, 1),
            (RawTokenKind::Star, _) => (TokenKind::Star, 1),
            (RawTokenKind::Slash, _) => (TokenKind::Slash, 1),
            (RawTokenKind::Percent, _) => (TokenKind::Percent, 1),
            (RawTokenKind::And, _) => (TokenKind::And, 1),

            (RawTokenKind::Unknown, _) => {
                errors.push(LexError {
                    span,
                    kind: LexErrorKind::UnknownCharacter,
                });
                index += 1;
                continue;
            }
            (RawTokenKind::Eof, _) => unreachable!("tokenize never yields Eof"),
        };

        let span = if consumed == 2 {
            span.to(raw[index + 1].1)
        } else {
            span
        };
        tokens.push(Token { kind: cooked, span });
        index += consumed;
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(pos, pos),
    });
    (tokens, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::kw;

    /// Renders each token as `Kind lo..hi`, resolving symbols so expectations
    /// stay readable.
    fn render(text: &str) -> Vec<String> {
        let mut interner = Interner::new();
        let (tokens, _) = lex(text, &mut interner);
        tokens
            .iter()
            .map(|token| {
                let body = match token.kind {
                    TokenKind::Ident(symbol) => format!("Ident({})", interner.get(symbol)),
                    TokenKind::Int(symbol) => format!("Int({})", interner.get(symbol)),
                    TokenKind::Lifetime(symbol) => format!("Lifetime({})", interner.get(symbol)),
                    other => format!("{other:?}"),
                };
                format!("{body} {}..{}", token.span.lo().0, token.span.hi().0)
            })
            .collect()
    }

    fn errors(text: &str) -> Vec<LexError> {
        let mut interner = Interner::new();
        lex(text, &mut interner).1
    }

    #[test]
    fn trivia_is_dropped_and_positions_are_absolute() {
        assert_eq!(
            render("fn main() { }"),
            [
                "Ident(fn) 0..2",
                "Ident(main) 3..7",
                "OpenParen 7..8",
                "CloseParen 8..9",
                "OpenBrace 10..11",
                "CloseBrace 12..13",
                "Eof 13..13",
            ],
        );
    }

    #[test]
    fn a_declaration_lexes_with_its_positions() {
        assert_eq!(
            render("let x: u32 = 1;"),
            [
                "Ident(let) 0..3",
                "Ident(x) 4..5",
                "Colon 5..6",
                "Ident(u32) 7..10",
                "Eq 11..12",
                "Int(1) 13..14",
                "Semi 14..15",
                "Eof 15..15",
            ],
        );
    }

    #[test]
    fn compound_operators_are_glued() {
        assert_eq!(
            render("a == b"),
            ["Ident(a) 0..1", "EqEq 2..4", "Ident(b) 5..6", "Eof 6..6"],
        );
        assert_eq!(render("-> !="), ["RArrow 0..2", "Ne 3..5", "Eof 5..5"]);
    }

    /// Adjacency is the whole point of gluing here rather than in the parser.
    #[test]
    fn gluing_does_not_cross_whitespace() {
        assert_eq!(
            render("a = = b"),
            [
                "Ident(a) 0..1",
                "Eq 2..3",
                "Eq 4..5",
                "Ident(b) 6..7",
                "Eof 7..7",
            ],
        );
    }

    /// Gluing is greedy from the left, so a third `&` stands alone.
    #[test]
    fn gluing_is_greedy_from_the_left() {
        assert_eq!(
            render("a &&& b"),
            [
                "Ident(a) 0..1",
                "AndAnd 2..4",
                "And 4..5",
                "Ident(b) 6..7",
                "Eof 7..7",
            ],
        );
    }

    #[test]
    fn a_lifetime_keeps_its_name_without_the_apostrophe() {
        assert_eq!(
            render("&'a mut x"),
            [
                "And 0..1",
                "Lifetime(a) 1..3",
                "Ident(mut) 4..7",
                "Ident(x) 8..9",
                "Eof 9..9",
            ],
        );
    }

    /// Keywords are identifiers whose symbol is a keyword; see the module
    /// documentation.
    #[test]
    fn keywords_arrive_as_identifiers_carrying_keyword_symbols() {
        let mut interner = Interner::new();
        let (tokens, _) = lex("fn while notakeyword", &mut interner);

        assert_eq!(tokens[0].kind, TokenKind::Ident(kw::FN));
        assert_eq!(tokens[1].kind, TokenKind::Ident(kw::WHILE));
        match tokens[2].kind {
            TokenKind::Ident(symbol) => assert!(!symbol.is_keyword()),
            other => panic!("expected an identifier, got {other:?}"),
        }
    }

    #[test]
    fn the_same_name_interns_to_the_same_symbol() {
        let mut interner = Interner::new();
        let (tokens, _) = lex("x + x", &mut interner);
        assert_eq!(tokens[0].kind, tokens[2].kind);
    }

    /// The end-to-end check on the position arithmetic: every span recovers the
    /// text it was lexed from.
    #[test]
    fn spans_slice_back_to_their_source() {
        let text = "fn take(r: &'a mut u32) -> u32 { 1_000 }";
        let mut interner = Interner::new();
        let (tokens, found) = lex(text, &mut interner);
        assert!(found.is_empty(), "unexpected errors: {found:?}");

        for token in &tokens {
            match token.kind {
                TokenKind::Ident(symbol) | TokenKind::Int(symbol) => {
                    assert_eq!(token.span.slice(text), &*interner.get(symbol));
                }
                TokenKind::Lifetime(symbol) => {
                    assert_eq!(token.span.slice(text), format!("'{}", interner.get(symbol)));
                }
                TokenKind::Eof => assert!(token.span.is_empty()),
                _ => assert!(!token.span.is_empty()),
            }
        }
    }

    #[test]
    fn an_unknown_character_is_reported_and_skipped() {
        assert_eq!(
            errors("a # b"),
            [LexError {
                span: Span::new(BytePos(2), BytePos(3)),
                kind: LexErrorKind::UnknownCharacter,
            }],
        );
        // Lexing continues, because the parser needs the remaining tokens.
        assert_eq!(
            render("a # b"),
            ["Ident(a) 0..1", "Ident(b) 4..5", "Eof 5..5"],
        );
    }

    #[test]
    fn an_unterminated_block_comment_is_reported() {
        assert_eq!(
            errors("x /* oops"),
            [LexError {
                span: Span::new(BytePos(2), BytePos(9)),
                kind: LexErrorKind::UnterminatedBlockComment,
            }],
        );
    }

    #[test]
    fn empty_input_yields_only_eof() {
        assert_eq!(render(""), ["Eof 0..0"]);
    }
}
