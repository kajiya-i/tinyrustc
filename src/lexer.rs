//! The raw lexer: the first of two lexing layers, mirroring `rustc_lexer`.
//!
//! This layer answers exactly one question - "what is the next lexical unit, and
//! how many bytes long is it" - and deliberately knows nothing else. It has no
//! notion of source files, byte offsets, symbols, keywords, or diagnostics. The
//! second layer, which attaches spans and interns identifiers, is built on top.
//!
//! Three consequences of that split are worth stating, because each is
//! surprising in isolation:
//!
//! - **Keywords do not exist here.** `fn` and `let` are [`RawTokenKind::Ident`]
//!   like any other word. Deciding that a particular identifier is a keyword
//!   requires an interner, which belongs to the layer above.
//! - **Compound operators are not glued.** `==` lexes as two [`RawTokenKind::Eq`]
//!   tokens and `->` as [`RawTokenKind::Minus`] followed by
//!   [`RawTokenKind::Gt`]. Joining them is the parser's job, because whether
//!   `>>` closes two generic parameters or shifts right is not a lexical
//!   question.
//! - **Whitespace and comments are tokens.** Nothing is silently dropped, so the
//!   lengths of the tokens produced for an input sum exactly to that input's
//!   length. That invariant is what makes byte offsets computed by the layer
//!   above trustworthy, and it is checked by the tests below.
//!
//! Lengths are `u32`, following `rustc`'s `BytePos`. This caps a single source
//! file at four gibibytes, a limit `rustc` also accepts.

use std::str::Chars;

/// Returned by [`Cursor::first`] when there is no character left.
///
/// A sentinel avoids `Option` in the hot path. It is only ever observed by
/// predicates, which are always paired with an end-of-input check, so a literal
/// NUL byte in the source is still lexed as [`RawTokenKind::Unknown`].
const EOF_CHAR: char = '\0';

/// A lexical unit: what it is, and how many bytes it spans.
///
/// Carries no position. The layer above knows where lexing started and
/// accumulates these lengths into absolute offsets.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct RawToken {
    pub kind: RawTokenKind,
    pub len: u32,
}

/// The kinds of lexical unit this layer can produce.
///
/// Punctuation is one variant per character by design; see the module
/// documentation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RawTokenKind {
    /// Trivia. Reported rather than skipped so that token lengths tile the input.
    Whitespace,
    LineComment,
    /// `terminated` is false for a black comment running to end of input. The
    /// token is still produced, because refusing to produce one would break the
    /// tiling invariant and leave the layer above unable to place the error.
    BlockComment {
        terminated: bool,
    },

    // Units spanning more than one character.
    Ident,
    Int,
    /// `'a`. Distinguished lexically because it is a lexical shape, unlike a
    /// keyword, which is not.
    Lifetime,

    // Single-character punctuation.
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    Comma,
    Colon,
    Semi,
    Dot,
    Eq,
    Bang,
    Lt,
    Gt,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    And,

    /// A character this language has no use for. Reported rather than skipped,
    /// leaving the decision of how loudly to complain to the layer above.
    Unknown,

    /// Input exhausted. Never yielded by [`tokenize`], which ends instead.
    Eof,
}

/// Whether `c` may begin an identifier.
///
/// `rustc` uses the Unicode XID properties here; this settles for
/// [`char::is_alphabetic`], which accepts the same ASCII and rejects nothing the
/// tests care about.
fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

/// Whether `c` may continue an identifier.
fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// A peekable, position-tracking reader over one string.
pub struct Cursor<'a> {
    /// Bytes left when the current token started, used to derive its length.
    len_remaining: usize,
    chars: Chars<'a>,
}

impl<'a> Cursor<'a> {
    pub fn new(input: &'a str) -> Cursor<'a> {
        Cursor {
            len_remaining: input.len(),
            chars: input.chars(),
        }
    }

    /// The next character without consuming it, or [`EOF_CHAR`].
    fn first(&self) -> char {
        self.chars.clone().next().unwrap_or(EOF_CHAR)
    }

    fn bump(&mut self) -> Option<char> {
        self.chars.next()
    }

    fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }

    /// Bytes consumed since the current token started.
    fn pos_within_token(&self) -> u32 {
        (self.len_remaining - self.chars.as_str().len()) as u32
    }

    fn reset_pos_within_token(&mut self) {
        self.len_remaining = self.chars.as_str().len();
    }

    /// Consumes characters while `predicate` holds.
    fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        while predicate(self.first()) && !self.is_eof() {
            self.bump();
        }
    }

    /// Consumes one token.
    ///
    /// Returns [`RawTokenKind::Eof`] with length zero once the input is
    /// exhausted, and keeps doing so if called again.
    pub fn advance_token(&mut self) -> RawToken {
        let Some(first) = self.bump() else {
            return RawToken {
                kind: RawTokenKind::Eof,
                len: 0,
            };
        };

        let kind = match first {
            c if c.is_whitespace() => {
                self.eat_while(char::is_whitespace);
                RawTokenKind::Whitespace
            }

            // The only place this layer needs to look ahead: a slash may open a
            // comment or stand alone as division.
            '/' => match self.first() {
                '/' => self.line_comment(),
                '*' => self.block_comment(),
                _ => RawTokenKind::Slash,
            },

            '\'' => self.lifetime(),

            c if is_ident_start(c) => {
                self.eat_while(is_ident_continue);
                RawTokenKind::Ident
            }

            c if c.is_ascii_digit() => {
                self.eat_while(|c| c.is_ascii_digit() || c == '_');
                RawTokenKind::Int
            }

            '(' => RawTokenKind::OpenParen,
            ')' => RawTokenKind::CloseParen,
            '{' => RawTokenKind::OpenBrace,
            '}' => RawTokenKind::CloseBrace,
            ',' => RawTokenKind::Comma,
            ':' => RawTokenKind::Colon,
            ';' => RawTokenKind::Semi,
            '.' => RawTokenKind::Dot,
            '=' => RawTokenKind::Eq,
            '!' => RawTokenKind::Bang,
            '<' => RawTokenKind::Lt,
            '>' => RawTokenKind::Gt,
            '+' => RawTokenKind::Plus,
            '-' => RawTokenKind::Minus,
            '*' => RawTokenKind::Star,
            '%' => RawTokenKind::Percent,
            '&' => RawTokenKind::And,

            _ => RawTokenKind::Unknown,
        };

        let len = self.pos_within_token();
        self.reset_pos_within_token();
        RawToken { kind, len }
    }

    /// Lexes from the second '/' to just before the newline, which belongs to
    /// the following whitespace token.
    fn line_comment(&mut self) -> RawTokenKind {
        self.bump();
        self.eat_while(|c| c != '\n');
        RawTokenKind::LineComment
    }

    /// Lexes from the `*` to the matching `*/`. honouring nesting as Rust does.
    fn block_comment(&mut self) -> RawTokenKind {
        self.bump();
        let mut depth = 1usize;
        while let Some(c) = self.bump() {
            match c {
                '/' if self.first() == '*' => {
                    self.bump();
                    depth += 1;
                }
                '*' if self.first() == '/' => {
                    self.bump();
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        RawTokenKind::BlockComment {
            terminated: depth == 0,
        }
    }

    /// Lexes a lifetime, the leading `'` already consumed.
    ///
    /// A bare `'` is [`RawTokenKind::Unkown`]. This language has no character
    /// literals, which is what makes the decision this simple; `rustc` has to
    /// look further ahead.
    fn lifetime(&mut self) -> RawTokenKind {
        if !is_ident_start(self.first()) {
            return RawTokenKind::Unknown;
        }
        self.eat_while(is_ident_continue);
        RawTokenKind::Lifetime
    }
}

/// Lexes `input`, yielding every token including trivia, and stopping at the
/// end rather than yielding [`RawTokenKind::Eof`].
pub fn tokenize(input: &str) -> impl Iterator<Item = RawToken> + '_ {
    let mut cursor = Cursor::new(input);
    std::iter::from_fn(move || {
        let token = cursor.advance_token();
        if token.kind == RawTokenKind::Eof {
            None
        } else {
            Some(token)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(input: &str) -> Vec<RawTokenKind> {
        tokenize(input).map(|token| token.kind).collect()
    }

    /// The invariant everything above this layer relies on: tokens tile the
    /// input exactly, so lengths can be accumulated into byte offsets.
    #[track_caller]
    fn assert_tiles(input: &str) {
        let total: u32 = tokenize(input).map(|token| token.len).sum();
        assert_eq!(total as usize, input.len(), "tokens must tile {input:?}");
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert_eq!(kinds(""), vec![]);
    }

    #[test]
    fn tokens_tile_their_input() {
        assert_tiles("");
        assert_tiles("   ");
        assert_tiles("fn main() { let x: u32 = 1; }");
        assert_tiles("// trailing comment");
        assert_tiles("/* unterminated");
        assert_tiles("let r: &'a mut u32 = &mut y;");
        assert_tiles("\u{3042}\u{3044} = 1;");
        assert_tiles("1_000 % 7");
    }

    /// Keyword recognition belongs to the layer above, so `fn` and `let` must
    /// arrive here as ordinary identifiers.
    #[test]
    fn keywords_lex_as_identifiers() {
        assert_eq!(kinds("fn"), vec![RawTokenKind::Ident]);
        assert_eq!(kinds("let"), vec![RawTokenKind::Ident]);
        assert_eq!(kinds("struct"), vec![RawTokenKind::Ident]);
    }

    /// Gluing is the parser's job, so compound operators arrive unassembled.
    #[test]
    fn compound_operators_are_not_glued() {
        assert_eq!(kinds("=="), vec![RawTokenKind::Eq, RawTokenKind::Eq]);
        assert_eq!(kinds("->"), vec![RawTokenKind::Minus, RawTokenKind::Gt]);
        assert_eq!(kinds("&&"), vec![RawTokenKind::And, RawTokenKind::And]);
    }

    #[test]
    fn identifiers_may_contain_digits_and_underscores() {
        assert_eq!(kinds("_x1"), vec![RawTokenKind::Ident]);
        assert_eq!(kinds("__"), vec![RawTokenKind::Ident]);
    }

    #[test]
    fn integers_admit_separators() {
        assert_eq!(kinds("1_000"), vec![RawTokenKind::Int]);
    }

    /// An identifier may not start with a digit, so this is two tokens rather
    /// than one error. Rejecting it is the parser's decision.
    #[test]
    fn digit_then_letters_is_two_tokens() {
        assert_eq!(kinds("1abc"), vec![RawTokenKind::Int, RawTokenKind::Ident]);
    }

    #[test]
    fn slash_alone_is_division() {
        assert_eq!(kinds("a / b").first(), Some(&RawTokenKind::Ident));
        assert_eq!(kinds("/"), vec![RawTokenKind::Slash]);
    }

    /// The newline is whitespace, not part of the comment.
    #[test]
    fn line_comment_stops_before_the_newline() {
        assert_eq!(
            kinds("// hi\nx"),
            vec![
                RawTokenKind::LineComment,
                RawTokenKind::Whitespace,
                RawTokenKind::Ident,
            ],
        );
    }

    #[test]
    fn block_comments_nest() {
        assert_eq!(
            kinds("/* /* */ */"),
            vec![RawTokenKind::BlockComment { terminated: true }],
        );
    }

    /// An unterminated comment still produces a token, flagged, so that the
    /// tiling invariant holds and the layer above can report it in place.
    #[test]
    fn unterminated_block_comment_is_flagged() {
        assert_eq!(
            kinds("/* nope"),
            vec![RawTokenKind::BlockComment { terminated: false }],
        );
    }

    #[test]
    fn lifetimes_are_lexed() {
        assert_eq!(kinds("'a"), vec![RawTokenKind::Lifetime]);
        assert_eq!(kinds("'_"), vec![RawTokenKind::Lifetime]);
    }

    #[test]
    fn a_bare_quote_is_unknown() {
        assert_eq!(kinds("'"), vec![RawTokenKind::Unknown]);
    }

    #[test]
    fn unrecognised_characters_are_reported_not_skipped() {
        assert_eq!(kinds("#"), vec![RawTokenKind::Unknown]);
        assert_eq!(kinds("$"), vec![RawTokenKind::Unknown]);
    }
}
