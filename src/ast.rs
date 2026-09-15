//! The abstract syntax tree.
//!
//! Corresponds to `rustc_ast`. Every node carries a [`Span`], because a node
//! that cannot point at source text cannot be reported on.
//!
//! # Error nodes
//!
//! [`ExprKind::Err`] stands where the parser could not build a real node. This
//! matters more than it looks: without it, a syntax error would have to abort
//! parsing, and one missing semicolon would hide every other diagnostic in the
//! file. `rustc` carries the same variant for the same reason, and it is what
//! lets later passes run on partially broken input.
//!
//! # Divergence from `rustc`
//!
//! - There is no node for parentheses. `rustc` keeps `ExprKind::Paren` so it can
//!   reproduce the source form; a batch compiler that never prints code back has
//!   no use for it, so grouping only widens the inner node's span.
//! - Nodes carry no identifier yet. `rustc` puts a `NodeId` on every AST node,
//!   which name resolution turns into `DefId`s and `HirId`s. That arrives when
//!   resolution needs it, rather than now.

use crate::span::Span;
use crate::symbol::Symbol;

/// An expression together with the source range it covers
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

/// Boxed because expressions nest, so the type is recursive.
///
/// `rustc` allocates AST nodes in an arena and threads `&'a Expr` instead, which
/// removes one pointer chase per node. That arrives with the type interner.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExprKind {
    /// An integer literal, still as written; see [`crate::token::TokenKind::Int`]
    Int(Symbol),
    /// `true` or `false`.
    ///
    /// Recognised here rather than in the lexer, because `true` arrives as an
    /// identifier whose synbol happens to be a keyword.
    Bool(bool),
    /// A bare name. Qualified paths do not exist in this subset yet.
    Path(Symbol),
    Unary(UnOp, Box<Expr>),
    /// `&x` or `&mut x`.
    ///
    /// Separate from [`ExprKind::Unary`] because the mutability is part of the
    /// operator, and because borrows are what the borrow checker will look for.
    Borrow {
        mutable: bool,
        operand: Box<Expr>,
    },
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Field {
        base: Box<Expr>,
        name: Symbol,
    },
    /// A node the parser could not build. See the module documentation.
    Err,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnOp {
    /// `-`
    Neg,
    /// `!`
    Not,
    /// `*`
    Deref,
}

impl UnOp {
    /// The operator as written, for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
            UnOp::Deref => "*",
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `&&`
    And,
}

impl BinOp {
    /// The operator as written, for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
        }
    }
}
