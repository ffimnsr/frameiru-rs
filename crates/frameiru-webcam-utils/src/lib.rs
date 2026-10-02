//! Frameiru webcam utilities.
//!
//! Linux CLI tooling for the Anker PowerConf C200 webcam. The camera exposes
//! vendor-only controls (FOV preset, HDR, horizontal flip, vertical screen,
//! anti-flicker) through a UVC Extension Unit, plus standard V4L2 controls.
//!
//! Control paths were reverse engineered from the official Anker app; see the
//! README of `erans/anker-powerconf-c200-linux-tools` for protocol details.

pub mod controls;
pub mod fov;
pub mod v4l2;
pub mod vendor;

use std::io;
use std::path::PathBuf;

use v4l2::V4l2Error;
use vendor::VendorError;

/// Top-level error type for the crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to open {path}")]
    OpenFailed { path: PathBuf, source: io::Error },
    #[error(transparent)]
    Vendor(#[from] VendorError),
    #[error(transparent)]
    V4l2(#[from] V4l2Error),
    #[error("unknown control: {name}")]
    UnknownControl { name: String },
    #[error("invalid value for {name}: {value}")]
    InvalidValue { name: String, value: String },
}
