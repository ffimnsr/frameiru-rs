//! Times one segmentation call on a synthetic half-white frame and prints
//! per-mask latency plus foreground fraction (sanity check for mask quality).
//!
//! Usage: `cargo run --release -p frameiru-segment --example time_segment
//! --features onnx -- <model.onnx> [WxH]` (size defaults to 1024x1024)

use frameiru_core::format::{FrameMetadata, PixelFormat};
use frameiru_core::{FrameBuffer, Resolution};
use frameiru_segment::{load_model, OnnxConfig};

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let size = std::env::args()
        .nth(2)
        .map(|s| {
            let (w, h) = s.split_once('x').expect("WxH");
            Resolution {
                width: w.parse().unwrap(),
                height: h.parse().unwrap(),
            }
        })
        .unwrap_or(Resolution {
            width: 1024,
            height: 1024,
        });
    let config = OnnxConfig::new(size).unwrap();
    let mut seg = load_model(&path, config).expect("load");
    let mut frame = FrameBuffer::new(FrameMetadata {
        sequence: 0,
        timestamp_us: 0,
        resolution: size,
        format: PixelFormat::Rgb8,
    });
    frame.data = vec![128u8; (size.area() * 3) as usize];
    for y in 0..(size.height / 2) as usize {
        for x in 0..size.width as usize {
            let i = (y * size.width as usize + x) * 3;
            frame.data[i] = 255;
            frame.data[i + 1] = 255;
            frame.data[i + 2] = 255;
        }
    }
    let t = std::time::Instant::now();
    match seg.segment(&frame) {
        Ok(mask) => {
            let el = t.elapsed();
            let fg = mask.data.iter().filter(|&&v| v > 0.5).count() as f64 / mask.data.len() as f64;
            println!(
                "one segment: {:?}, fg fraction {:.3}, mask {}x{}",
                el, fg, mask.resolution.width, mask.resolution.height
            );
        }
        Err(e) => println!("segment failed after {:?}: {e}", t.elapsed()),
    }
}
