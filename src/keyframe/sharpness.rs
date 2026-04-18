//! Tenengrad sharpness detector.
//!
//! Computes the Tenengrad focus measure — the mean of squared 3×3 Sobel
//! gradient magnitudes on a luma plane:
//!
//! ```text
//! T = (1 / N_inner) · Σ (gx² + gy²)
//! ```
//!
//! over all **interior** pixels (i.e. excluding the 1-pixel border where
//! the 3×3 kernel cannot fit). Higher values mean sharper frames; the
//! absolute scale depends on input resolution, so scores are only
//! comparable within a sequence of frames at the same dimensions. In
//! the selection pipeline we compare frames from the same shot (same
//! downscaled dims from [`crate::keyframe::preprocess::Downscaler`]), so
//! this is sufficient.
//!
//! Two entry points:
//!
//! - [`Detector::observe_luma`]: runs the full pipeline — Sobel + reduction.
//! - [`Detector::observe_sobel`]: skips the Sobel stage, consuming
//!   caller-supplied `gx` and `gy` planes. Useful when another component
//!   (e.g. a future Canny front-end or gradient-based detector) already
//!   computes the same Sobel output.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{LumaFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::sharpness::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//!
//! # let bytes = vec![0u8; 256 * 144];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let luma = LumaFrame::new(&bytes, 256, 144, 256, Timestamp::new(0, tb));
//! let sharpness = det.observe_luma(luma);
//! assert!(sharpness >= 0.0);
//! ```

use crate::frame::LumaFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the sharpness detector.
///
/// Currently only carries the `use_simd` flag for forward-compatibility;
/// the scalar path is always used in v0.1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
}

impl Default for Options {
  fn default() -> Self {
    Self { use_simd: true }
  }
}

impl Options {
  /// Creates a new [`Options`] matching [`Options::default`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self { use_simd: true }
  }

  /// Sets whether to dispatch to SIMD backends when available.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, on: bool) -> Self {
    self.use_simd = on;
    self
  }

  /// Returns whether SIMD dispatch is currently enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn use_simd(&self) -> bool {
    self.use_simd
  }
}

/// Pure-algo state machine that reduces a luma frame to its Tenengrad
/// sharpness score.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self { opts }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes Tenengrad sharpness on `luma` and returns the score.
  /// Frames narrower or shorter than 3 pixels have no interior and
  /// yield `0.0`.
  ///
  /// The caller already carries the frame's timestamp via
  /// [`LumaFrame::timestamp`] — it is not re-emitted here.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32 {
    crate::arch::tenengrad(
      luma.data(),
      luma.width() as usize,
      luma.height() as usize,
      luma.stride() as usize,
      self.opts.use_simd,
    )
  }

  /// Skips the Sobel stage of [`Self::observe_luma`] — the caller
  /// already has the two gradient planes and just wants the Tenengrad
  /// reduction `mean(gx² + gy²)`.
  ///
  /// `gx` and `gy` are tight-packed `width × height` planes. Row `y`
  /// starts at element index `y * width`. Values are `i16` because 3×3
  /// Sobel on 8-bit luma produces output in `[-1020, 1020]` — this fits
  /// comfortably, and the i16 layout is cache-friendly.
  ///
  /// Border pixels (the first/last row and first/last column) are
  /// excluded from the reduction, matching [`Self::observe_luma`]'s
  /// interior-pixel average. On equivalent inputs the two entry points
  /// return identical scores modulo float rounding. Frames narrower or
  /// shorter than 3 pixels yield `0.0`.
  ///
  /// # Panics
  ///
  /// In debug builds: if `gx.len() < width * height` or
  /// `gy.len() < width * height`. Release builds trust the caller.
  pub fn observe_sobel(
    &mut self,
    gx: &[i16],
    gy: &[i16],
    width: usize,
    height: usize,
  ) -> f32 {
    tenengrad_from_gradients(gx, gy, width, height)
  }
}

/// Scalar Tenengrad reduction over pre-computed Sobel gradient planes.
/// Border pixels are excluded — same interior-pixel definition as
/// [`tenengrad_scalar`].
fn tenengrad_from_gradients(gx: &[i16], gy: &[i16], w: usize, h: usize) -> f32 {
  debug_assert!(gx.len() >= w * h, "gx plane length mismatch");
  debug_assert!(gy.len() >= w * h, "gy plane length mismatch");

  if w < 3 || h < 3 {
    return 0.0;
  }
  let interior = (w - 2) * (h - 2);

  // Per-pixel max |gx·gy| at 3×3 Sobel on 8-bit = 1020² ≈ 1.04 M. i32
  // handles the product; i64 handles the sum across any realistic
  // interior count.
  let mut acc: i64 = 0;
  for y in 1..h - 1 {
    for x in 1..w - 1 {
      let idx = y * w + x;
      let gxv = gx[idx] as i32;
      let gyv = gy[idx] as i32;
      acc += (gxv * gxv + gyv * gyv) as i64;
    }
  }

  ((acc as f64) / (interior as f64)) as f32
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{LumaFrame, Timebase, Timestamp};
  use core::num::NonZeroU32;
  use std::vec;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn timestamp() -> Timestamp {
    Timestamp::new(99, Timebase::new(1, nz(1000)))
  }

  fn tight_luma(data: &[u8], w: u32, h: u32) -> LumaFrame<'_> {
    LumaFrame::new(data, w, h, w, timestamp())
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn options_builder_roundtrips() {
    let o = Options::new().with_simd(false);
    assert!(!o.use_simd());
  }

  #[test]
  fn uniform_frame_has_zero_sharpness() {
    let data = vec![128u8; 32 * 32];
    let mut det = Detector::new(Options::default());
    let s = det.observe_luma(tight_luma(&data, 32, 32));
    assert_eq!(s, 0.0);
  }

  #[test]
  fn too_small_frame_returns_zero() {
    // 2×2 has no interior pixels.
    let data = vec![0u8, 255, 255, 0];
    let mut det = Detector::new(Options::default());
    let s = det.observe_luma(tight_luma(&data, 2, 2));
    assert_eq!(s, 0.0);
  }

  #[test]
  fn vertical_edge_matches_expected_tenengrad() {
    // 16×16 frame, left half 0, right half 255.
    // At the x=7 and x=8 columns of interior pixels (0-indexed), the
    // 3×3 Sobel sees the black/white boundary → gx = 4·255 = 1020,
    // gy = 0, so per-pixel (gx² + gy²) = 1_040_400.
    // Interior rows: y ∈ [1, 14], so 14 rows × 2 columns = 28 hot
    // pixels, contributing 28 × 1_040_400 = 29_131_200.
    // Interior count = 14 × 14 = 196.
    // Tenengrad = 29_131_200 / 196 = 148_628.571…
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let score = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    let expected = 29_131_200.0_f32 / 196.0;
    assert!(
      (score - expected).abs() < 1.0,
      "expected ~{expected}, got {score}"
    );
  }

  #[test]
  fn horizontal_edge_matches_vertical_edge() {
    // Same score as vertical edge on a symmetric frame.
    let (w, h) = (16usize, 16usize);
    let mut v_data = vec![0u8; w * h];
    let mut h_data = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        v_data[y * w + x] = 255;
      }
    }
    for y in 8..h {
      for x in 0..w {
        h_data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let sv = det.observe_luma(tight_luma(&v_data, w as u32, h as u32));
    let sh = det.observe_luma(tight_luma(&h_data, w as u32, h as u32));
    assert!((sv - sh).abs() < 1.0);
  }

  #[test]
  fn sharper_frame_scores_higher() {
    // Soft ramp (low-frequency content) vs. hard edge (high-frequency).
    let (w, h) = (32usize, 32usize);
    let mut ramp = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        ramp[y * w + x] = ((x * 255) / (w - 1)) as u8;
      }
    }
    let mut edge = vec![0u8; w * h];
    for y in 0..h {
      for x in w / 2..w {
        edge[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let s_ramp = det.observe_luma(tight_luma(&ramp, w as u32, h as u32));
    let s_edge = det.observe_luma(tight_luma(&edge, w as u32, h as u32));
    assert!(
      s_edge > s_ramp,
      "edge frame ({s_edge}) should be sharper than ramp ({s_ramp})"
    );
  }

  #[test]
  fn stride_padding_is_ignored() {
    // 8×4 pixels, stride 16. Padding filled with 255 (would be a
    // screaming edge on column x=w if it leaked into the kernel).
    let w = 8usize;
    let h = 4usize;
    let stride = 16usize;
    let mut data = vec![255u8; stride * h];
    // Uniform gray in pixel area → expected sharpness 0.
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 100;
      }
    }
    let f = LumaFrame::new(&data, w as u32, h as u32, stride as u32, timestamp());
    let mut det = Detector::new(Options::default());
    let s = det.observe_luma(f);
    assert_eq!(s, 0.0, "padding bytes leaked into Sobel kernel");
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![0u8; 16 * 16];
    let s = det.observe_luma(tight_luma(&data, 16, 16));
    assert_eq!(s, 0.0);
  }

  // ---- observe_sobel --------------------------------------------------------

  /// Builds tight-packed `gx` and `gy` planes by running the same 3×3
  /// Sobel the detector uses internally. Used to compare the luma-path
  /// and sobel-path outputs on identical inputs.
  fn make_sobel_planes(src: &[u8], w: usize, h: usize) -> (std::vec::Vec<i16>, std::vec::Vec<i16>) {
    let mut gx = vec![0i16; w * h];
    let mut gy = vec![0i16; w * h];
    for y in 1..h.saturating_sub(1) {
      for x in 1..w.saturating_sub(1) {
        let p = |dy: isize, dx: isize| -> i32 {
          src[((y as isize + dy) as usize) * w + ((x as isize + dx) as usize)] as i32
        };
        let tl = p(-1, -1);
        let t = p(-1, 0);
        let tr = p(-1, 1);
        let l = p(0, -1);
        let r = p(0, 1);
        let bl = p(1, -1);
        let b = p(1, 0);
        let br = p(1, 1);
        let vx = -tl - 2 * l - bl + tr + 2 * r + br;
        let vy = -tl - 2 * t - tr + bl + 2 * b + br;
        gx[y * w + x] = vx as i16;
        gy[y * w + x] = vy as i16;
      }
    }
    (gx, gy)
  }

  #[test]
  fn observe_sobel_zero_gradients_yield_zero() {
    let (w, h) = (8usize, 8usize);
    let gx = vec![0i16; w * h];
    let gy = vec![0i16; w * h];
    let mut det = Detector::new(Options::default());
    let s = det.observe_sobel(&gx, &gy, w, h);
    assert_eq!(s, 0.0);
  }

  #[test]
  fn observe_sobel_matches_observe_luma_on_equivalent_input() {
    // Vertical edge — same fixture used in vertical_edge_matches_expected.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        data[y * w + x] = 255;
      }
    }
    let (gx, gy) = make_sobel_planes(&data, w, h);

    let mut det = Detector::new(Options::default());
    let from_luma = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    let from_sobel = det.observe_sobel(&gx, &gy, w, h);

    assert!(
      (from_luma - from_sobel).abs() < 1e-3,
      "luma path {from_luma} != sobel path {from_sobel}"
    );
  }

  #[test]
  fn observe_sobel_too_small_returns_zero() {
    let gx = vec![100i16; 4];
    let gy = vec![100i16; 4];
    let mut det = Detector::new(Options::default());
    assert_eq!(det.observe_sobel(&gx, &gy, 2, 2), 0.0);
  }

  #[test]
  fn observe_sobel_uniform_nonzero_interior() {
    // 4×4 frame. Interior is the 2×2 centre block. Fill gx=gy=100
    // across the whole plane; interior sum = 4 × (100² + 100²) = 80,000;
    // divide by interior count 4 → Tenengrad = 20,000.
    let (w, h) = (4usize, 4usize);
    let gx = vec![100i16; w * h];
    let gy = vec![100i16; w * h];
    let mut det = Detector::new(Options::default());
    let s = det.observe_sobel(&gx, &gy, w, h);
    assert!((s - 20_000.0).abs() < 1.0, "expected ~20000, got {s}");
  }
}
