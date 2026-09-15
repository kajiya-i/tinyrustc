//! A recursive-descent parser with Pratt precedence for expressions.
//!
//! Corresponds to `rustc_parse`. Two invariants hold everything together:
//!
//! - **The cursor never runs off the end.** [`crate::token::Lexed`] always ends
//!   in [`TokenKind::Eof`] and `Parser::advance` refuses to step past it, so
//!   there is no bounds check anywhere below.
//! - **Recovery always consumes.** A path that reports an error without moving
//!   the cursor turns any surrounding loop into an infinite one. Every error in
//!   `Parser::primary` is paired with an `advance`.
//!
//! # Precedence
//!
//! Binary operators are handled by precedence climbing rather than one function
//! per level, which is what `rustc` does in `parse_assoc_expr_with`. Adding an
//! operator is then a line in `binding_power` instead of a new function.
//!
//! # Divergence from `rustc`
//!
//! Comparison operators are left-associative here. Rust makes them
//! non-associative, so `a < b < c` is a syntax error rather than
//! `(a < b) < c`. Reporting that needs one more check and is not yet worth it.

use crate::ast::{BinOp, Expr, ExprKind, UnOp};
use crate::span::Span;
use crate::symbol::{Symbol, kw};
use crate::token::{Token, TokenKind};

/// A syntax error, ready to be turned into a diagnostic.
///
/// FIXME: the message is a `String`. `rustc` carries structured diagnostics with
/// error codes and suggestions, which is what makes them translatable and
/// machine-applicable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

/// The binding power of `kind` as an infix operator, if it is one.
///
/// Higher binds tighter. The table is the whole precedence specification.
fn binding_power(kind: TokenKind) -> Option<(BinOp, u8)> {
    let pair = match kind {
        TokenKind::AndAnd => (BinOp::And, 1),
        TokenKind::EqEq => (BinOp::Eq, 2),
        TokenKind::Ne => (BinOp::Ne, 2),
        TokenKind::Lt => (BinOp::Lt, 2),
        TokenKind::Le => (BinOp::Le, 2),
        TokenKind::Gt => (BinOp::Gt, 2),
        TokenKind::Ge => (BinOp::Ge, 2),
        TokenKind::Plus => (BinOp::Add, 3),
        TokenKind::Minus => (BinOp::Sub, 3),
        TokenKind::Star => (BinOp::Mul, 4),
        TokenKind::Slash => (BinOp::Div, 4),
        TokenKind::Percent => (BinOp::Rem, 4),
        _ => return None,
    };
    Some(pair)
}

/// Parses `tokens` as a single expression, reporting anything left over.
pub fn parse_expr(tokens: &[Token]) -> (Expr, Vec<ParseError>) {
    let mut parser = Parser::new(tokens);
    let expr = parser.expr();
    if parser.peek() != TokenKind::Eof {
        let span = parser.peek_span();
        parser.error(span, "unexpected token after expression");
    }
    (expr, parser.errors)
}

/// A cursor over a token stream, accumulating errors as it goes.
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    errors: Vec<ParseError>,
}

impl<'a> Parser<'a> {
    /// # Panics
    ///
    /// Panics in debug builds if `tokens` does not end in [`TokenKind::Eof`].
    /// Every method below relies on that sentinel instead of a bounds check.
    pub fn new(tokens: &'a [Token]) -> Parser<'a> {
        debug_assert!(
            matches!(tokens.last().map(|token| token.kind), Some(TokenKind::Eof)),
            "the token stream must end in Eof",
        );
        Parser {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    fn current(&self) -> Token {
        self.tokens[self.pos]
    }

    fn peek(&self) -> TokenKind {
        self.current().kind
    }

    fn peek_span(&self) -> Span {
        self.current().span
    }

    /// Consumes and returns the current token, stopping at [`TokenKind::Eof`].
    ///
    /// Refusing to advance past `Eof` is what keeps [`Parser::current`] in
    /// bounds without a check.
    fn advance(&mut self) -> Token {
        let token = self.current();
        if token.kind != TokenKind::Eof {
            self.pos += 1;
        }
        token
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes `kind` and returns its span, or reports `what` as missing and
    /// returns the span of whatever was there instead.
    ///
    /// Deliberately does not consume on failure: the token that was found is
    /// usually meaningful to the caller further up.
    fn expect(&mut self, kind: TokenKind, what: &str) -> Span {
        let span = self.peek_span();
        if self.peek() == kind {
            self.advance();
        } else {
            self.error(span, format!("expected {what}"));
        }
        span
    }

    fn eat_keyword(&mut self, keyword: Symbol) -> bool {
        self.eat(TokenKind::Ident(keyword))
    }

    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.errors.push(ParseError {
            span,
            message: message.into(),
        });
    }

    pub fn expr(&mut self) -> Expr {
        self.binary(0)
    }

    /// Precedence climbing. `min_power` is the lowest binding power this call
    /// will accept as an infix operator.
    fn binary(&mut self, min_power: u8) -> Expr {
        let mut lhs = self.unary();

        while let Some((op, power)) = binding_power(self.peek()) {
            if power < min_power {
                break;
            }
            self.advance();
            // `power + 1` makes the operator left-associative: the recursive
            // call refuses an operator of equal power, so it comes back and the
            // loop folds it into the left side instead.
            let rhs = self.binary(power + 1);
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                kind: ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }

        lhs
    }

    fn unary(&mut self) -> Expr {
        let start = self.peek_span();

        // `&` is handled separately because the mutability that may follow is
        // part of the operator.
        if self.peek() == TokenKind::And {
            self.advance();
            let mutable = self.eat_keyword(kw::MUT);
            let operand = self.unary();
            let span = start.to(operand.span);
            return Expr {
                kind: ExprKind::Borrow {
                    mutable,
                    operand: Box::new(operand),
                },
                span,
            };
        }

        let op = match self.peek() {
            TokenKind::Minus => UnOp::Neg,
            TokenKind::Bang => UnOp::Not,
            TokenKind::Star => UnOp::Deref,
            _ => return self.postfix(),
        };
        self.advance();
        let operand = self.unary();
        let span = start.to(operand.span);
        Expr {
            kind: ExprKind::Unary(op, Box::new(operand)),
            span,
        }
    }

    /// Calls and field accesses, which bind tighter than any prefix operator and
    /// chain left to right.
    fn postfix(&mut self) -> Expr {
        let mut expr = self.primary();

        loop {
            match self.peek() {
                TokenKind::OpenParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if self.peek() != TokenKind::CloseParen {
                        args.push(self.expr());
                        while self.eat(TokenKind::Comma) {
                            // A trailing comma is allowed, as in Rust.
                            if self.peek() == TokenKind::CloseParen {
                                break;
                            }
                            args.push(self.expr());
                        }
                    }
                    let end = self.expect(TokenKind::CloseParen, "`)`");
                    let span = expr.span.to(end);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    let token = self.advance();
                    let span = expr.span.to(token.span);
                    expr = match token.kind {
                        TokenKind::Ident(name) if !name.is_keyword() => Expr {
                            kind: ExprKind::Field {
                                base: Box::new(expr),
                                name,
                            },
                            span,
                        },
                        _ => {
                            self.error(token.span, "expected a field name");
                            Expr {
                                kind: ExprKind::Err,
                                span,
                            }
                        }
                    };
                }
                _ => break,
            }
        }

        expr
    }

    fn primary(&mut self) -> Expr {
        let token = self.current();

        match token.kind {
            TokenKind::Int(symbol) => {
                self.advance();
                Expr {
                    kind: ExprKind::Int(symbol),
                    span: token.span,
                }
            }
            TokenKind::Ident(symbol) if symbol == kw::TRUE => {
                self.advance();
                Expr {
                    kind: ExprKind::Bool(true),
                    span: token.span,
                }
            }
            TokenKind::Ident(symbol) if symbol == kw::FALSE => {
                self.advance();
                Expr {
                    kind: ExprKind::Bool(false),
                    span: token.span,
                }
            }
            TokenKind::Ident(symbol) if !symbol.is_keyword() => {
                self.advance();
                Expr {
                    kind: ExprKind::Path(symbol),
                    span: token.span,
                }
            }
            TokenKind::OpenParen => {
                let open = self.advance().span;
                let inner = self.expr();
                let close = self.expect(TokenKind::CloseParen, "`)`");
                // Grouping widens the span and leaves the tree alone; see the
                // ast module documentation.
                Expr {
                    kind: inner.kind,
                    span: open.to(close),
                }
            }
            _ => {
                self.error(token.span, "expected an expression");
                // Consuming is what guarantees progress. `advance` is a no-op at
                // Eof, which is safe because no loop above continues on Eof.
                self.advance();
                Expr {
                    kind: ExprKind::Err,
                    span: token.span,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Interner;
    use crate::token::lex;

    /// Renders an expression as an S-expression, so precedence is visible at a
    /// glance and expectations stay short.
    fn render(expr: &Expr, interner: &Interner) -> String {
        match &expr.kind {
            ExprKind::Int(symbol) | ExprKind::Path(symbol) => interner.get(*symbol).to_string(),
            ExprKind::Bool(value) => value.to_string(),
            ExprKind::Unary(op, operand) => {
                let name = match op {
                    UnOp::Neg => "neg",
                    UnOp::Not => "not",
                    UnOp::Deref => "deref",
                };
                format!("({name} {})", render(operand, interner))
            }
            ExprKind::Borrow { mutable, operand } => {
                let name = if *mutable { "&mut" } else { "&" };
                format!("({name} {})", render(operand, interner))
            }
            ExprKind::Binary(op, lhs, rhs) => format!(
                "({} {} {})",
                op.as_str(),
                render(lhs, interner),
                render(rhs, interner),
            ),
            ExprKind::Call { callee, args } => {
                let mut out = format!("(call {}", render(callee, interner));
                for arg in args {
                    out.push(' ');
                    out.push_str(&render(arg, interner));
                }
                out.push(')');
                out
            }
            ExprKind::Field { base, name } => {
                format!("(field {} {})", render(base, interner), interner.get(*name))
            }
            ExprKind::Err => "<err>".to_string(),
        }
    }

    /// The rendered tree only.
    fn tree(text: &str) -> String {
        let mut interner = Interner::new();
        let lexed = lex(text, &mut interner);
        let (expr, errors) = parse_expr(&lexed.tokens);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        render(&expr, &interner)
    }

    /// The rendered tree and the error messages, for recovery tests.
    fn tree_and_errors(text: &str) -> (String, Vec<String>) {
        let mut interner = Interner::new();
        let lexed = lex(text, &mut interner);
        let (expr, errors) = parse_expr(&lexed.tokens);
        let messages = errors.into_iter().map(|error| error.message).collect();
        (render(&expr, &interner), messages)
    }

    #[test]
    fn a_literal_parses_to_itself() {
        assert_eq!(tree("1"), "1");
        assert_eq!(tree("1_000"), "1_000");
        assert_eq!(tree("true"), "true");
        assert_eq!(tree("false"), "false");
        assert_eq!(tree("x"), "x");
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(tree("a + b * c"), "(+ a (* b c))");
        assert_eq!(tree("a * b + c"), "(+ (* a b) c)");
        assert_eq!(tree("a % b * c"), "(* (% a b) c)");
    }

    #[test]
    fn arithmetic_is_left_associative() {
        assert_eq!(tree("a - b - c"), "(- (- a b) c)");
        assert_eq!(tree("a / b / c"), "(/ (/ a b) c)");
    }

    #[test]
    fn comparison_binds_tighter_than_conjunction() {
        assert_eq!(tree("a && b == c"), "(&& a (== b c))");
        assert_eq!(tree("a < b && c"), "(&& (< a b) c)");
    }

    #[test]
    fn arithmetic_binds_tighter_than_comparison() {
        assert_eq!(tree("a + b < c * d"), "(< (+ a b) (* c d))");
    }

    #[test]
    fn parentheses_override_precedence() {
        assert_eq!(tree("(a + b) * c"), "(* (+ a b) c)");
        assert_eq!(tree("a * (b + c)"), "(* a (+ b c))");
    }

    #[test]
    fn prefix_operators_bind_tighter_than_any_infix() {
        assert_eq!(tree("-a + b"), "(+ (neg a) b)");
        assert_eq!(tree("*p + 1"), "(+ (deref p) 1)");
        assert_eq!(tree("!a && b"), "(&& (not a) b)");
    }

    #[test]
    fn prefix_operators_nest() {
        assert_eq!(tree("- - a"), "(neg (neg a))");
        assert_eq!(tree("!*p"), "(not (deref p))");
    }

    #[test]
    fn a_borrow_carries_its_mutability() {
        assert_eq!(tree("&x"), "(& x)");
        assert_eq!(tree("&mut x"), "(&mut x)");
        assert_eq!(tree("&x + 1"), "(+ (& x) 1)");
    }

    #[test]
    fn calls_and_fields_chain_left_to_right() {
        assert_eq!(tree("f()"), "(call f)");
        assert_eq!(tree("f(a, b)"), "(call f a b)");
        assert_eq!(tree("f(a,)"), "(call f a)");
        assert_eq!(tree("a.b.c"), "(field (field a b) c)");
        assert_eq!(tree("f(a).b"), "(field (call f a) b)");
        assert_eq!(tree("a.b(c)"), "(call (field a b) c)");
    }

    #[test]
    fn call_arguments_are_full_expressions() {
        assert_eq!(tree("f(a + b, *c)"), "(call f (+ a b) (deref c))");
    }

    // ---- Recovery ----

    /// A missing operand becomes an error node, so the surrounding tree survives
    /// for later passes.
    #[test]
    fn a_missing_operand_becomes_an_error_node() {
        let (rendered, errors) = tree_and_errors("a +");
        assert_eq!(rendered, "(+ a <err>)");
        assert_eq!(errors, ["expected an expression"]);
    }

    /// Empty input terminates rather than looping, because `advance` is a no-op
    /// at Eof and no loop continues on Eof.
    #[test]
    fn empty_input_reports_once_and_terminates() {
        let (rendered, errors) = tree_and_errors("");
        assert_eq!(rendered, "<err>");
        assert_eq!(errors, ["expected an expression"]);
    }

    /// The leading operator is consumed, so the parser makes progress and then
    /// notices the leftover token.
    #[test]
    fn a_leading_infix_operator_is_reported_twice() {
        let (rendered, errors) = tree_and_errors("+ a");
        assert_eq!(rendered, "<err>");
        assert_eq!(
            errors,
            [
                "expected an expression",
                "unexpected token after expression"
            ],
        );
    }

    #[test]
    fn an_unclosed_call_is_reported() {
        let (rendered, errors) = tree_and_errors("f(a");
        assert_eq!(rendered, "(call f a)");
        assert_eq!(errors, ["expected `)`"]);
    }

    #[test]
    fn a_keyword_cannot_start_an_expression() {
        let (_, errors) = tree_and_errors("let");
        assert_eq!(errors, ["expected an expression"]);
    }

    #[test]
    fn a_keyword_cannot_be_a_field_name() {
        let (rendered, errors) = tree_and_errors("a.let");
        assert_eq!(rendered, "<err>");
        assert_eq!(errors, ["expected a field name"]);
    }

    /// Pins a known limitation. The lexer glues `&&` into one token, so a double
    /// reference cannot be parsed. `rustc` splits glued operators back apart in
    /// the parser where the grammar demands it; doing so here needs a token
    /// stream that can be re-split mid-parse.
    #[test]
    fn a_double_reference_is_not_yet_parsable() {
        let (rendered, errors) = tree_and_errors("&&x");
        assert_eq!(rendered, "<err>");
        assert_eq!(
            errors,
            [
                "expected an expression",
                "unexpected token after expression"
            ],
        );
    }

    // ---- Spans ----

    #[test]
    fn a_nodes_span_covers_all_of_its_children() {
        let text = "a + b * c";
        let mut interner = Interner::new();
        let lexed = lex(text, &mut interner);
        let (expr, _) = parse_expr(&lexed.tokens);
        assert_eq!(expr.span.slice(text), text);
    }

    #[test]
    fn parentheses_are_included_in_the_span() {
        let text = "(a + b)";
        let mut interner = Interner::new();
        let lexed = lex(text, &mut interner);
        let (expr, _) = parse_expr(&lexed.tokens);
        assert_eq!(expr.span.slice(text), text);
    }
}
