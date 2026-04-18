//! Color-space conversions for packed 24-bit BGR video frames.
//!
//! # Byte order
//!
//! All functions in this module consume **BGR** — three bytes per pixel,
//! in the order `B, G, R`. This matches the convention used by the
//! `content` detector and OpenCV's default layout. Callers holding RGB
//! buffers must swizzle `R` and `B` before calling (cheap — a single
//! SIMD shuffle per 16 pixels on any modern target).
//!
//! # Dispatch
//!
//! SIMD is selected per-call via the `use_simd` flag and, on x86 with
//! `std`, further dispatched at runtime via `is_x86_feature_detected!`.
//! `use_simd == false` forces the scalar fallback — useful for tests and
//! for environments where vector units are throttled.
//!
//! Current backends:
//!
//! - aarch64 NEON, x86 SSSE3 / AVX2, wasm `simd128`, and a scalar
//!   fallback for [`bgr_to_hsv_planes`].
//! - Scalar only (for now) for [`bgr_to_luma`]; SIMD lands in a later
//!   revision.

/// Converts a packed 24-bit BGR frame into three planar 8-bit HSV buffers
/// matching OpenCV's `cv2.COLOR_BGR2HSV` semantics (H in `[0, 179]`, S and
/// V in `[0, 255]`).
///
/// # Arguments
///
/// - `h_out`, `s_out`, `v_out`: destination planes, each at least
///   `width * height` bytes. Row `y` of each plane starts at byte offset
///   `y * width`.
/// - `src`: source BGR buffer; must be at least `stride * height` bytes.
///   Row `y` starts at byte offset `y * stride`; within each row, pixel
///   `x` occupies bytes `x*3 .. x*3 + 3` in the order `B, G, R`.
/// - `width`, `height`: frame dimensions in pixels.
/// - `stride`: source row stride in bytes (must satisfy `stride >= 3 *
///   width`).
/// - `use_simd`: `true` to dispatch to the best available SIMD backend;
///   `false` forces the scalar fallback.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) {
  crate::arch::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, use_simd);
}

/// Converts a packed 24-bit BGR frame to a single-plane 8-bit BT.601 luma
/// buffer.
///
/// Uses the standard BT.601 coefficients via the fixed-point approximation
///
/// ```text
/// Y = (77·R + 150·G + 29·B) >> 8
/// ```
///
/// which rounds to within one least-significant bit of the floating-point
/// expression `Y = 0.299·R + 0.587·G + 0.114·B`. The coefficients sum to
/// exactly 256, so the output is always in `[0, 255]`.
///
/// # Arguments
///
/// - `out`: destination luma plane; must be at least `width * height`
///   bytes. Row `y` starts at byte offset `y * width`.
/// - `src`: source BGR buffer; must be at least `stride * height` bytes.
///   Row `y` starts at byte offset `y * stride`; within each row, pixel
///   `x` occupies bytes `x*3 .. x*3 + 3` in the order `B, G, R`.
/// - `width`, `height`: frame dimensions in pixels.
/// - `stride`: source row stride in bytes (must satisfy `stride >= 3 *
///   width`).
/// - `use_simd`: `true` to dispatch to the best available SIMD backend
///   (aarch64 NEON today; x86 SSSE3/AVX2 and wasm-simd128 in follow-up
///   commits); `false` forces the scalar fallback.
///
/// # Panics (debug builds)
///
/// Debug builds verify `stride >= 3 * width`, `src.len() >= stride *
/// height`, and `out.len() >= width * height`. In release builds the
/// caller is trusted — the frame-type constructors in this crate
/// (`RgbFrame::try_new`, `LumaFrame::try_new`) already validate these
/// invariants and should be preferred as the construction path.
#[inline]
pub fn bgr_to_luma(
  out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) {
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;

  debug_assert!(s >= w.saturating_mul(3), "stride must be >= width*3");
  debug_assert!(src.len() >= s.saturating_mul(h), "src too short");
  debug_assert!(out.len() >= w.saturating_mul(h), "out too short");

  crate::arch::bgr_to_luma(out, src, width, height, stride, use_simd);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use std::vec;

  fn make_bgr(w: usize, h: usize, seed: u32) -> std::vec::Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    let mut rng = seed;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  #[test]
  fn bgr_to_luma_black_is_zero() {
    let (w, h) = (8, 4);
    let src = vec![0u8; w * h * 3];
    let mut out = vec![255u8; w * h];
    bgr_to_luma(&mut out, &src, w as u32, h as u32, (w * 3) as u32, false);
    assert!(out.iter().all(|&y| y == 0), "black frame should be all zero");
  }

  #[test]
  fn bgr_to_luma_white_is_saturated() {
    let (w, h) = (8, 4);
    let src = vec![255u8; w * h * 3];
    let mut out = vec![0u8; w * h];
    bgr_to_luma(&mut out, &src, w as u32, h as u32, (w * 3) as u32, false);
    // 77 + 150 + 29 = 256; (256*255) >> 8 = 255.
    assert!(out.iter().all(|&y| y == 255), "white frame should saturate");
  }

  #[test]
  fn bgr_to_luma_coefficients_match_bt601() {
    // Pure red, pure green, pure blue — each should land near its BT.601 weight.
    // Pixel layout is BGR.
    let red_bgr = [0u8, 0, 255];
    let green_bgr = [0u8, 255, 0];
    let blue_bgr = [255u8, 0, 0];

    let mut out = [0u8; 1];

    bgr_to_luma(&mut out, &red_bgr, 1, 1, 3, false);
    // 77 * 255 / 256 = 76
    assert_eq!(out[0], 76);

    bgr_to_luma(&mut out, &green_bgr, 1, 1, 3, false);
    // 150 * 255 / 256 = 149
    assert_eq!(out[0], 149);

    bgr_to_luma(&mut out, &blue_bgr, 1, 1, 3, false);
    // 29 * 255 / 256 = 28
    assert_eq!(out[0], 28);
  }

  #[test]
  fn bgr_to_luma_honors_stride_padding() {
    // 4×2 pixels, 4 bytes of padding per row (stride 16 for pixel row = 12).
    let (w, h) = (4usize, 2usize);
    let stride = 16usize;
    let mut src = vec![0u8; stride * h];
    // Set first row to white, second row to black, leave padding as 0.
    for x in 0..w {
      src[x * 3] = 255;
      src[x * 3 + 1] = 255;
      src[x * 3 + 2] = 255;
    }
    let mut out = vec![0u8; w * h];
    bgr_to_luma(&mut out, &src, w as u32, h as u32, stride as u32, false);
    assert!(out[..w].iter().all(|&y| y == 255));
    assert!(out[w..].iter().all(|&y| y == 0));
  }

  #[test]
  fn bgr_to_hsv_planes_reexport_works() {
    // Smoke-test the re-export — full tests live in `crate::arch`.
    let (w, h) = (16, 4);
    let src = make_bgr(w, h, 0x9E3779B9);
    let mut ho = vec![0u8; w * h];
    let mut so = vec![0u8; w * h];
    let mut vo = vec![0u8; w * h];
    bgr_to_hsv_planes(
      &mut ho,
      &mut so,
      &mut vo,
      &src,
      w as u32,
      h as u32,
      (w * 3) as u32,
      false,
    );
    assert!(vo.iter().any(|&v| v > 0));
  }
}
