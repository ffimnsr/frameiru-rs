//! Pixel format decoding helpers.
//!
//! YUYV (Y'CbCr 4:2:2) -> RGB24 uses the standard ITU-R BT.601 limited range
//! matrix with fixed-point coefficients, so roundtrips against
//! `frameiru_sink::convert::rgb8_to_yuyv422` are lossy only in rounding.

use frameiru_core::error::FrameiruError;

#[cfg(feature = "mjpeg")]
use frameiru_core::format::Resolution;

/// Decodes a tightly packed YUYV 4:2:2 frame into RGB24.
///
/// `input` must hold `width * height * 2` bytes, `output` at least
/// `width * height * 3` bytes. Chroma samples are shared between pixel pairs
/// (4:2:2 subsampling); each pair emits `[Y0, U, Y1, V]`.
pub fn yuyv_to_rgb8(
    input: &[u8],
    width: u32,
    height: u32,
    output: &mut [u8],
) -> Result<(), FrameiruError> {
    let w = width as usize;
    let h = height as usize;
    if width == 0 || height == 0 {
        return Err(FrameiruError::InvalidArgument(
            "YUYV dimensions must be non-zero".into(),
        ));
    }
    let need_in = w * h * 2;
    let need_out = w * h * 3;
    if input.len() < need_in {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_in,
            have: input.len(),
        });
    }
    if output.len() < need_out {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_out,
            have: output.len(),
        });
    }

    // Even widths (the real camera case) go through the SIMD-accelerated
    // `yuv` crate; odd widths keep the scalar loop because our packed rows
    // truncate the trailing pair to [Y, U] rather than padding it.
    if width.is_multiple_of(2) {
        let packed = yuv::YuvPackedImage {
            yuy: input,
            yuy_stride: w as u32 * 2,
            width,
            height,
        };
        yuv::yuyv422_to_rgb(
            &packed,
            output,
            w as u32 * 3,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt601,
        )
        .map_err(|e| FrameiruError::InvalidArgument(format!("yuv decode failed: {e}")))?;
        return Ok(());
    }

    for row in 0..h {
        let in_row = &input[row * w * 2..(row + 1) * w * 2];
        let out_row = &mut output[row * w * 3..(row + 1) * w * 3];
        for x in 0..w {
            let pair_base = (x / 2) * 4;
            let y0 = in_row[pair_base] as i32;
            let u = in_row[pair_base + 1] as i32;
            // Odd widths truncate the final pair to [Y0, U]: reuse chroma.
            let v = if pair_base + 3 < in_row.len() {
                in_row[pair_base + 3] as i32
            } else {
                u
            };

            // BT.601 limited range inverse matrix (fixed point, /256).
            let c = y0 - 16;
            let d = u - 128;
            let e = v - 128;
            let r = (298 * c + 409 * e + 128) >> 8;
            let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            let b = (298 * c + 516 * d + 128) >> 8;

            let i = x * 3;
            out_row[i] = clamp_u8(r);
            out_row[i + 1] = clamp_u8(g);
            out_row[i + 2] = clamp_u8(b);
        }
    }
    Ok(())
}

/// Decodes a tightly packed NV12 (4:2:0 bi-planar) frame into RGB24.
///
/// `input` holds the full-resolution Y plane followed by interleaved UV at
/// half resolution (`w * h * 3 / 2` bytes for even dimensions). Uses the
/// `yuv` crate's SIMD NV12 path (its fastest conversion). Odd dimensions
/// are rejected: 4:2:0 chroma subsampling requires even pairs.
pub fn nv12_to_rgb8(
    input: &[u8],
    width: u32,
    height: u32,
    output: &mut [u8],
) -> Result<(), FrameiruError> {
    let w = width as usize;
    let h = height as usize;
    if width == 0 || height == 0 {
        return Err(FrameiruError::InvalidArgument(
            "NV12 dimensions must be non-zero".into(),
        ));
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(FrameiruError::InvalidArgument(
            "NV12 dimensions must be even".into(),
        ));
    }
    let need_in = w * h * 3 / 2;
    let need_out = w * h * 3;
    if input.len() < need_in {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_in,
            have: input.len(),
        });
    }
    if output.len() < need_out {
        return Err(FrameiruError::InsufficientCapacity {
            need: need_out,
            have: output.len(),
        });
    }

    let y_plane = &input[..w * h];
    let uv_plane = &input[w * h..];
    let planar = yuv::YuvBiPlanarImage {
        y_plane,
        y_stride: width,
        uv_plane,
        uv_stride: width,
        width,
        height,
    };
    yuv::yuv_nv12_to_rgb(
        &planar,
        output,
        width * 3,
        yuv::YuvRange::Limited,
        yuv::YuvStandardMatrix::Bt601,
        yuv::YuvConversionMode::Balanced,
    )
    .map_err(|e| FrameiruError::InvalidArgument(format!("nv12 decode failed: {e}")))?;
    Ok(())
}

/// Decodes a JPEG (MJPEG video frame) into tightly packed RGB24.
///
/// Returns the decoded frame resolution and resizes `output` to hold the
/// pixels. Only compiled when the `mjpeg` feature is enabled.
#[cfg(feature = "mjpeg")]
pub fn mjpeg_to_rgb8(input: &[u8], output: &mut Vec<u8>) -> Result<Resolution, FrameiruError> {
    use zune_jpeg::JpegDecoder;

    let mut decoder = JpegDecoder::new(input);
    let pixels = decoder
        .decode()
        .map_err(|e| FrameiruError::Unsupported(format!("mjpeg decode failed: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| FrameiruError::Unsupported("mjpeg decode produced no image info".into()))?;
    let resolution = Resolution::new(info.width as u32, info.height as u32)?;
    let need = resolution.area() as usize * 3;
    output.resize(need, 0);
    if pixels.len() < need {
        return Err(FrameiruError::InsufficientCapacity {
            need,
            have: pixels.len(),
        });
    }
    output.copy_from_slice(&pixels[..need]);
    Ok(resolution)
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_frame_decodes_to_gray() {
        let w = 4u32;
        let h = 2u32;
        // Limited-range mid gray: Y=126, U=V=128.
        let mut input = vec![0u8; (w * h * 2) as usize];
        for row in 0..h {
            for x in 0..w {
                let p = ((row * w + x) * 2) as usize;
                input[p] = 126;
                // Chroma is shared per pixel pair: write it once per pair.
                if x % 2 == 0 {
                    input[p + 1] = 128;
                    input[p + 3] = 128;
                }
            }
        }
        let mut out = vec![0u8; (w * h * 3) as usize];
        yuyv_to_rgb8(&input, w, h, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v as i32 - 128).abs() <= 6));
    }

    #[test]
    fn checks_buffer_sizes() {
        let mut out = vec![0u8; 3];
        assert!(matches!(
            yuyv_to_rgb8(&[0u8; 1], 4, 2, &mut out),
            Err(FrameiruError::InsufficientCapacity { .. })
        ));
        assert!(yuyv_to_rgb8(&[0u8; 16], 4, 2, &mut [0u8; 8]).is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        let mut out = vec![0u8; 3 * 3];
        assert!(yuyv_to_rgb8(&[0u8; 18], 0, 3, &mut out).is_err());
    }

    #[test]
    fn nv12_gray_frame_decodes_to_gray() {
        // 4x2 NV12: 8 luma bytes then 4 interleaved UV bytes (all 128).
        let w = 4u32;
        let h = 2u32;
        let mut input = vec![0u8; (w * h * 3 / 2) as usize];
        input[..(w * h) as usize].fill(126); // limited-range mid gray
        input[(w * h) as usize..].fill(128); // neutral chroma
        let mut out = vec![0u8; (w * h * 3) as usize];
        nv12_to_rgb8(&input, w, h, &mut out).unwrap();
        assert!(out.iter().all(|&v| (v as i32 - 128).abs() <= 6));
    }

    #[test]
    fn nv12_rejects_odd_dimensions_and_small_buffers() {
        assert!(nv12_to_rgb8(&[0u8; 8], 3, 2, &mut [0u8; 18]).is_err());
        assert!(nv12_to_rgb8(&[0u8; 4], 4, 2, &mut [0u8; 24]).is_err());
    }

    #[test]
    fn odd_width_is_supported() {
        // 3x1 frame: needs 6 in bytes, 9 out bytes.
        let input = vec![126u8, 128, 126, 128, 126, 128];
        let mut out = vec![0u8; 9];
        assert!(yuyv_to_rgb8(&input, 3, 1, &mut out).is_ok());
    }
}
