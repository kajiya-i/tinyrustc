# Tiny Rust Compiler

## Learning from the rustc dev guide

This compiler is built alongside the [rustc dev guide](https://rustc-dev-guide.rust-lang.org/),
in roughly this reading order:

1. Overview / Query system
2. `DefId` and `HirId`
3. `TyCtxt` and arenas
4. MIR construction
5. **MIR borrow check** — the main goal
6. Incremental compilation

The guide is not read front to back before writing code. Implementation comes
first; when something gets stuck, the relevant chapter is read to unblock it,
then work continues. Read on demand, not up front.