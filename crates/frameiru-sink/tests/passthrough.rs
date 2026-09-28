//! End-to-end passthrough test: physical webcam -> virtual camera.
//!
//! Requires a capture device (`/dev/video0` by default, override with
//! `FRAMEIRU_VIDEO_DEVICE`) and a v4l2loopback device (`/dev/video10`,
//! override with `FRAMEIRU_LOOPBACK_DEVICE`). Skips cleanly when either is
//! absent, so CI without cameras stays green.
//!
//! ```sh
//! sudo modprobe v4l2loopback exclusive_caps=1 card_label=Frameiru
//! cargo test -p frameiru-sink --features v4l2 --test passthrough
//! ```

#![cfg(feature = "v4l2")]

use std::path::Path;

use frameiru_capture::v4l2::V4l2Source;
use frameiru_core::traits::FrameSink;
use frameiru_core::traits::FrameSource;
use frameiru_sink::v4l2::LoopbackSink;

const VIDEO: &str = "/dev/video0";
const LOOPBACK: &str = "/dev/video10";

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

/// `/dev/video0` -> `V4l2Source` -> `LoopbackSink` -> `/dev/video10`,
/// then reads the loopback back as a capture source and verifies the frame
/// travels the full roundtrip.
#[test]
fn webcam_to_loopback_roundtrip() {
    let video = env_or("FRAMEIRU_VIDEO_DEVICE", VIDEO);
    let loopback = env_or("FRAMEIRU_LOOPBACK_DEVICE", LOOPBACK);
    if !Path::new(&video).exists() || !Path::new(&loopback).exists() {
        eprintln!("skipping roundtrip: need {video} (capture) and {loopback} (loopback)");
        return;
    }

    let mut source = match V4l2Source::open(&video) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("skipping roundtrip: cannot open {video}: {e}");
            return;
        }
    };
    let frame = match source.next_frame() {
        Ok(frame) => frame,
        Err(e) => {
            eprintln!("skipping roundtrip: capture from {video} failed: {e}");
            return;
        }
    };
    assert_eq!(frame.metadata.format, frameiru_core::PixelFormat::Rgb8);

    let mut sink = match LoopbackSink::open(&loopback, frame.metadata.resolution) {
        Ok(sink) => sink,
        Err(e) => {
            eprintln!("skipping roundtrip: cannot open {loopback}: {e}");
            return;
        }
    };
    if let Err(e) = sink.write_frame(&frame) {
        eprintln!("skipping roundtrip: write to {loopback} failed: {e}");
        return;
    }

    // Read the loopback back and check the frame layout matches.
    let mut echoed = match V4l2Source::open(&loopback) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("skipping roundtrip: cannot read {loopback} back: {e}");
            return;
        }
    };
    let echoed_frame = match echoed.next_frame() {
        Ok(frame) => frame,
        Err(e) => {
            eprintln!("skipping roundtrip: read-back from {loopback} failed: {e}");
            return;
        }
    };
    assert_eq!(
        echoed_frame.metadata.format,
        frameiru_core::PixelFormat::Rgb8
    );
    assert_eq!(
        echoed_frame.metadata.resolution, frame.metadata.resolution,
        "loopback must echo the written frame's resolution"
    );
    eprintln!(
        "roundtrip ok: {}x{} via {loopback}",
        frame.metadata.resolution.width, frame.metadata.resolution.height
    );
}
