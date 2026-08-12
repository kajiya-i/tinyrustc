## Summary

<!-- What this PR does and why, in 1–4 sentences. -->

## Changes

<!-- List of notable changes. For bug fixes, include root cause and fix.
     For new features, include what was added. For tests, list what was covered. -->

## Related Issues

<!-- Use the keyword that fits:
     Closes #N  — feature implementation or task completion
     Fixes #N   — bug fix
     Resolves #N — general resolution (discussion, refactor, etc.) -->

## Test Plan

CI runs all of these; tick them once it is green, or after running them locally.

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
