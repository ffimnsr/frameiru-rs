//! Generated Slint component types.
//!
//! `slint::include_modules!()` can't be used: it expands to the single file
//! named by `SLINT_INCLUDE_GENERATED`, which the last `slint_build::compile`
//! call overwrites, so pull every output in manually.

include!(concat!(env!("OUT_DIR"), "/main.rs"));
include!(concat!(env!("OUT_DIR"), "/tray.rs"));
include!(concat!(env!("OUT_DIR"), "/mini.rs"));
