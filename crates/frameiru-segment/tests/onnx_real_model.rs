//! Ad-hoc integration test for real ONNX models: verifies the full
//! preprocess -> model -> postprocess chain preserves orientation and
//! geometry. Run with the model path in `FRAMEIRU_TEST_MODEL`:
//!
//! ```sh
//! FRAMEIRU_TEST_MODEL=/path/to/model.onnx cargo test -p frameiru-segment \
//!     --features onnx --test onnx_real_model -- --nocapture
//! ```

#![cfg(feature = "onnx")]

use frameiru_core::buffer::Mask;
use frameiru_core::format::{FrameMetadata, PixelFormat, Resolution};
use frameiru_core::traits::Segmenter;
use frameiru_core::FrameBuffer;
use frameiru_segment::{OnnxConfig, OnnxSegmenter};
use serial_test::serial;

const FRAME: (u32, u32) = (640, 480);

fn load() -> OnnxSegmenter {
    let path =
        std::env::var("FRAMEIRU_TEST_MODEL").unwrap_or_else(|_| "models/silueta.onnx".into());
    let config = OnnxConfig::new(Resolution {
        width: 320,
        height: 320,
    })
    .unwrap();
    match OnnxSegmenter::load(&path, config) {
        Ok(segmenter) => segmenter,
        Err(e) => {
            eprintln!("skipping: cannot load model {path}: {e}");
            std::process::exit(0);
        }
    }
}

fn frame_with_edge(edge_y: u32, top: (u8, u8, u8), bottom: (u8, u8, u8)) -> FrameBuffer {
    let (w, h) = FRAME;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        let (r, g, b) = if y < edge_y { top } else { bottom };
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
    }
    FrameBuffer {
        metadata: FrameMetadata {
            sequence: 0,
            timestamp_us: 0,
            resolution: Resolution {
                width: w,
                height: h,
            },
            format: PixelFormat::Rgb8,
        },
        data,
    }
}

/// Finds the first row whose mean mask value crosses `threshold`.
fn first_edge_row(mask: &Mask, threshold: f32) -> u32 {
    let w = mask.resolution.width as usize;
    for y in 0..mask.resolution.height as usize {
        let row = &mask.data[y * w..(y + 1) * w];
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        if mean > threshold {
            return y as u32;
        }
    }
    u32::MAX
}

#[test]
#[serial]
fn mask_orientation_matches_frame() {
    let mut segmenter = load();
    let (w, h) = FRAME;

    // Top 2/3 white, bottom 1/3 black: a unique edge at y = 320.
    let frame = frame_with_edge(320, (255, 255, 255), (0, 0, 0));
    let mask = segmenter.segment(&frame).expect("segmentation");
    assert_eq!(
        mask.resolution,
        Resolution {
            width: w,
            height: h
        }
    );
    let edge = first_edge_row(&mask, 0.5);

    println!("predicted edge row: {edge} (expected ~320, flipped would be ~160)");
    assert!(
        (240..=400).contains(&edge),
        "mask edge at row {edge}: model path appears vertically flipped"
    );
}

#[test]
#[serial]
fn mask_is_not_horizontally_mirrored() {
    let mut segmenter = load();
    let (w, h) = FRAME;

    // Left 2/3 white, right 1/3 black: a unique vertical edge at x = 427.
    let mut frame = frame_with_edge(0, (0, 0, 0), (0, 0, 0));
    for y in 0..h {
        for x in 0..(w * 2 / 3) {
            let i = ((y * w + x) * 3) as usize;
            frame.data[i] = 255;
            frame.data[i + 1] = 255;
            frame.data[i + 2] = 255;
        }
    }
    let mask = segmenter.segment(&frame).expect("segmentation");
    // First column crossing 0.5 going left -> right.
    let mut edge_x = u32::MAX;
    for x in 0..w {
        let col_mean: f32 = (0..h).map(|y| mask.data[(y * w + x) as usize]).sum::<f32>() / h as f32;
        if col_mean > 0.5 {
            edge_x = x;
            break;
        }
    }
    println!("predicted edge col: {edge_x} (expected ~427, mirrored would be ~213)");
    assert!(
        (340..=w).contains(&edge_x),
        "mask edge at col {edge_x}: model path appears horizontally mirrored"
    );
}

/// RVM recurrent-state flow: states start zeroed, roll forward per frame,
/// and reset to zero on `reset_state`. Run with `FRAMEIRU_TEST_MODEL`
/// pointing at `rvm_mobilenetv3_fp32.onnx`.
#[test]
#[serial]
fn rvm_state_flow_and_reset() {
    use frameiru_segment::{OnnxConfig, RvmSegmenter};

    let path = std::env::var("FRAMEIRU_TEST_MODEL").unwrap_or_default();
    let config = match OnnxConfig::new(Resolution {
        width: 256,
        height: 256,
    }) {
        Ok(c) => c,
        Err(_) => std::process::exit(0),
    };
    let mut rvm = match RvmSegmenter::load(&path, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping RVM test (point FRAMEIRU_TEST_MODEL at an RVM model): {e}");
            std::process::exit(0);
        }
    };

    assert_eq!(rvm.state_checksum(), 0, "states start zeroed");
    let frame = frame_with_edge(320, (255, 255, 255), (0, 0, 0));
    let mask = rvm.segment(&frame).expect("first RVM inference");
    assert_eq!(
        mask.resolution,
        Resolution {
            width: 640,
            height: 480
        }
    );
    assert_ne!(rvm.state_checksum(), 0, "states roll forward");

    rvm.reset_state();
    assert_eq!(rvm.state_checksum(), 0, "reset zeroes the states");
    // The pipeline keeps segmenting after a reset.
    rvm.segment(&frame).expect("inference after reset");
}

/// The embedded MediaPipe model must load from the binary and
/// produce a full-resolution mask in [0, 1] — the zero-setup default path.
#[test]
#[serial]
fn embedded_model_segments_a_frame() {
    use frameiru_segment::{load_embedded, EMBEDDED_MODEL_INPUT};
    let config = OnnxConfig::new(EMBEDDED_MODEL_INPUT).unwrap();
    let mut seg = load_embedded(config).expect("embedded model must load");
    assert_eq!(seg.input_resolution(), EMBEDDED_MODEL_INPUT);
    let frame = frame_with_edge(240, (255, 255, 255), (0, 0, 0));
    let mask = seg.segment(&frame).expect("segment with embedded model (full-frame pass)");
    assert_eq!(mask.resolution, frame.metadata.resolution);
    assert_eq!(mask.data.len(), (FRAME.0 * FRAME.1) as usize);
    assert!(
        mask.data.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "mask values must stay in [0, 1]"
    );

    // Consecutive frame: exercises Dynamic ROI Zoom tracking on the discovered subject
    let mask2 = seg.segment(&frame).expect("segment with embedded model (roi-zoom pass)");
    assert_eq!(mask2.resolution, frame.metadata.resolution);
    assert_eq!(mask2.data.len(), (FRAME.0 * FRAME.1) as usize);
    assert!(
        mask2.data.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "mask values must stay in [0, 1]"
    );
}
