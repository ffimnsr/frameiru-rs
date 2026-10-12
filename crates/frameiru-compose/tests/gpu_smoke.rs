//! GPU compositor smoke test: runs the composite modes and every overlay on a
//! real adapter and skips cleanly when none is available (CI-safe).
//! Requires `--features gpu`; without the feature this file is empty.
#![cfg(feature = "gpu")]

use frameiru_compose::GpuCompositor;
use frameiru_core::buffer::Mask;
use frameiru_core::format::{FrameMetadata, PixelFormat};
use frameiru_core::traits::Compositor;
use frameiru_core::{BackgroundMode, FrameBuffer, OverlayMode, Resolution};

fn res(w: u32, h: u32) -> Resolution {
    Resolution {
        width: w,
        height: h,
    }
}

fn frame(res: Resolution, fill: u8) -> FrameBuffer {
    let mut f = FrameBuffer::new(FrameMetadata {
        sequence: 0,
        timestamp_us: 0,
        resolution: res,
        format: PixelFormat::Rgb8,
    });
    f.data.fill(fill);
    f
}

fn mask(res: Resolution, fill: f32) -> Mask {
    Mask {
        resolution: res,
        data: vec![fill; res.area() as usize],
    }
}

/// Smoke test: runs composite modes on a real adapter, or skips when no
/// GPU/headless adapter is available (CI-safe).
#[test]
fn composites_on_available_adapter() {
    let mut gpu = match GpuCompositor::try_new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipping: no wgpu adapter available: {e}");
            return;
        }
    };
    gpu.update_background(BackgroundMode::Color { r: 255, g: 0, b: 0 })
        .unwrap();
    let src = frame(res(32, 24), 0);
    let m = mask(res(32, 24), 0.0); // fully background
    let mut out = FrameBuffer::new(FrameMetadata {
        sequence: 0,
        timestamp_us: 0,
        resolution: res(32, 24),
        format: PixelFormat::Rgb8,
    });
    gpu.composite(&src, &m, &mut out).unwrap();
    assert_eq!(out.data.len(), 32 * 24 * 3);
    // Color mode with mask 0: everything becomes the red background.
    assert!(out
        .data
        .chunks(3)
        .all(|p| p[0] == 255 && p[1] == 0 && p[2] == 0));

    // Passthrough copies the source regardless of mask.
    gpu.update_background(BackgroundMode::Passthrough).unwrap();
    let gray = frame(res(32, 24), 77);
    gpu.composite(&gray, &m, &mut out).unwrap();
    assert!(out.data.iter().all(|&v| v == 77));

    // Blur mode: top half white, bottom half black, mask 0. Vertical
    // asymmetry catches any NDC y-flip in the rendering path.
    gpu.update_background(BackgroundMode::Blur { radius: 1.0 })
        .unwrap();
    // Top half white, bottom half black, for every column.
    let mut src2 = frame(res(32, 24), 0);
    for y in 0..12usize {
        for x in 0..32usize {
            let i = (y * 32 + x) * 3;
            src2.data[i] = 255;
            src2.data[i + 1] = 255;
            src2.data[i + 2] = 255;
        }
    }
    gpu.composite(&src2, &m, &mut out).unwrap();
    // Radius 1 keeps the extremes: no vertical flip allowed.
    let top = out.data[0];
    let bottom = out.data[(23 * 32) * 3];
    assert!(
        top > 128 && bottom < 128,
        "vertical orientation broken: top={top} bottom={bottom}"
    );

    // Blur with a sharp subject: foreground pixels (alpha 1, left half)
    // must be excluded from the kernel so the bright side cannot smear
    // into the dark background (halo-free, U9.5).
    let mut src3 = frame(res(32, 24), 255);
    for row in src3.data.as_chunks_mut::<{ 32 * 3 }>().0 {
        for v in row[32 / 2 * 3..].iter_mut() {
            *v = 0;
        }
    }
    let mut m2 = mask(res(32, 24), 0.0);
    for y in 0..24usize {
        for x in 0..16usize {
            m2.data[y * 32 + x] = 1.0;
        }
    }
    gpu.update_background(BackgroundMode::Blur { radius: 4.0 })
        .unwrap();
    gpu.composite(&src3, &m2, &mut out).unwrap();
    for y in 0..24usize {
        for x in 18..32usize {
            let i = (y * 32 + x) * 3;
            assert_eq!(
                &out.data[i..i + 3],
                &[0, 0, 0],
                "gpu halo at row {y} col {x}"
            );
        }
    }
    // Subject side stays the source.
    assert_eq!(&out.data[..3], &[255, 255, 255]);

    // Scanlines overlay: odd rows darkened on top of a solid background.
    // (GPU textures are sRGB, so only relative comparisons are safe.)
    gpu.update_overlay(OverlayMode::Scanlines);
    gpu.update_background(BackgroundMode::Color { r: 0, g: 0, b: 0 })
        .unwrap();
    let src4 = frame(res(8, 4), 200);
    let m4 = mask(res(8, 4), 1.0);
    gpu.composite(&src4, &m4, &mut out).unwrap();
    assert_eq!(out.data[0], 200, "bright row must be untouched");
    assert!(
        out.data[2 * 8 * 3] < out.data[0],
        "scanline row must be darker, got {} vs {}",
        out.data[2 * 8 * 3],
        out.data[0]
    );

    // Scanlines overlay with Passthrough background: must darken scanline rows!
    gpu.update_background(BackgroundMode::Passthrough).unwrap();
    let src_pass = frame(res(8, 4), 200);
    let m_pass = mask(res(8, 4), 0.0);
    gpu.composite(&src_pass, &m_pass, &mut out).unwrap();
    assert_eq!(
        out.data[0], 200,
        "bright row untouched in passthrough scanlines"
    );
    assert!(
        out.data[2 * 8 * 3] < out.data[0],
        "scanline row must be darker in passthrough scanlines, got {} vs {}",
        out.data[2 * 8 * 3],
        out.data[0]
    );

    // LightLeak: check that center doesn't blow out to 255 on a dark frame
    gpu.update_overlay(OverlayMode::LightLeak);
    let mut src_leak = frame(res(64, 48), 0);
    src_leak.metadata.timestamp_us = 5_000_000;
    let m_leak = mask(res(64, 48), 0.0);
    gpu.composite(&src_leak, &m_leak, &mut out).unwrap();
    let center_r = out.data[(8 * 64 + 24) * 3];
    assert!(
        center_r < 180,
        "light leak must not blow out to white/255, got {center_r}"
    );
    assert!(
        center_r > 30,
        "light leak must visibly illuminate the blob center, got {center_r}"
    );

    // CRT: test in passthrough mode, corner must be darker than center
    gpu.update_overlay(OverlayMode::Crt);
    let src_crt = frame(res(32, 32), 255);
    let m_crt = mask(res(32, 32), 0.0);
    gpu.composite(&src_crt, &m_crt, &mut out).unwrap();
    let center_val = out.data[(16 * 32 + 16) * 3];
    let corner_val = out.data[0];
    assert!(
        corner_val < center_val,
        "CRT corner ({corner_val}) must be darker than center ({center_val})"
    );
    let bright_crt = out.data[(16 * 32 + 16) * 3];
    let scan_crt = out.data[(18 * 32 + 16) * 3];
    assert!(
        scan_crt < bright_crt,
        "CRT scanline row ({scan_crt}) must be darker than bright row ({bright_crt})"
    );

    // ChromaticAberration: radial RGB split on a hard vertical edge.
    gpu.update_overlay(OverlayMode::ChromaticAberration);
    let (cw, chh) = (256usize, 32usize);
    let mut src_ca = frame(res(cw as u32, chh as u32), 255);
    for y in 0..chh {
        for x in 0..8usize {
            let i = (y * cw + x) * 3;
            src_ca.data[i..i + 3].copy_from_slice(&[0, 0, 0]);
        }
    }
    let m_ca = mask(res(cw as u32, chh as u32), 0.0);
    gpu.composite(&src_ca, &m_ca, &mut out).unwrap();
    let edge = (chh / 2 * cw + 8) * 3;
    assert!(
        out.data[edge] > out.data[edge + 2] + 40,
        "chromatic split must separate red ({}) and blue ({}) at the edge",
        out.data[edge],
        out.data[edge + 2]
    );

    // VHS: the tracking bar starts at the top at t = 0, so its rows are
    // brighter than rows lower down on a uniform frame (row 0 -> top).
    gpu.update_overlay(OverlayMode::Vhs);
    let src_vhs = frame(res(32, 64), 200);
    let m_vhs = mask(res(32, 64), 0.0);
    gpu.composite(&src_vhs, &m_vhs, &mut out).unwrap();
    let in_bar = out.data[16 * 3];
    let out_of_bar = out.data[(32 * 32 + 16) * 3];
    assert!(
        in_bar > out_of_bar,
        "VHS tracking bar must brighten its rows: {in_bar} vs {out_of_bar}"
    );

    // Dither: 1-bit output, both tones present on a mid-gray frame.
    gpu.update_overlay(OverlayMode::Dither);
    let src_d = frame(res(32, 32), 128);
    let m_d = mask(res(32, 32), 0.0);
    gpu.composite(&src_d, &m_d, &mut out).unwrap();
    assert!(out.data.iter().all(|&v| v == 0 || v == 255));
    assert!(out.data.contains(&255), "dither must show both tones");

    // Halftone: dot centers lit, cell gaps dark on a mid-gray frame.
    gpu.update_overlay(OverlayMode::Halftone);
    gpu.composite(&src_d, &m_d, &mut out).unwrap();
    let hi = out.data.iter().copied().max().unwrap_or(0);
    let lo = out.data.iter().copied().min().unwrap_or(255);
    assert!(hi > 80, "halftone dots must be lit, max {hi}");
    assert!(lo < 20, "halftone gaps must stay dark, min {lo}");

    // Grayscale: all channels equal.
    gpu.update_overlay(OverlayMode::Grayscale);
    let mut src_g = frame(res(16, 16), 0);
    for px in src_g.data.as_chunks_mut::<3>().0 {
        px.copy_from_slice(&[200, 100, 50]);
    }
    let m_g = mask(res(16, 16), 0.0);
    gpu.composite(&src_g, &m_g, &mut out).unwrap();
    assert!(
        out.data.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]),
        "grayscale output must have equal channels"
    );

    // Cyberpunk: dark frame gains the blue/purple shadow grade and the
    // bright block feeds the halation bloom into its surroundings.
    gpu.update_overlay(OverlayMode::Cyberpunk);
    let mut src_cy = frame(res(32, 32), 20);
    for y in 2..6 {
        for x in 2..6 {
            let i = (y * 32 + x) * 3;
            src_cy.data[i..i + 3].copy_from_slice(&[255, 250, 235]);
        }
    }
    let m_cy = mask(res(32, 32), 0.0);
    gpu.composite(&src_cy, &m_cy, &mut out).unwrap();
    // Shadow push: dark pixels gain blue, matching the CPU grade.
    let dark = &out.data[(15 * 32 + 15) * 3..(15 * 32 + 15) * 3 + 3];
    assert!(
        dark[2] > dark[0],
        "cyberpunk shadows must drift blue: {dark:?}"
    );
    // Bloom: a pixel just outside the bright block is lifted above its
    // source tone by the blurred halo.
    let fringe = &out.data[(6 * 32 + 8) * 3..(6 * 32 + 8) * 3 + 3];
    assert!(fringe[1] > 20, "halo must brighten the fringe: {fringe:?}");
    gpu.update_overlay(OverlayMode::None);
}
