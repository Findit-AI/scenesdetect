//! Immerkaer fast noise estimator.
//!
//! Estimates per-pixel additive-Gaussian noise standard deviation σₙ
//! on a luma plane using Immerkaer's 1996 single-pass technique:
//!
//! ```text
//!         ⎡  1  -2   1 ⎤
//!     N = ⎢ -2   4  -2 ⎥
//!         ⎣  1  -2   1 ⎦
//!
//!     σₙ ≈ √(π/2) · (1 / (6·N_inner)) · Σ |luma ⊛ N|
//! ```
//!
//! Border pixels are excluded (matching the Tenengrad convention).
//! Higher values mean noisier frames; the absolute scale depends on
//! input resolution, so scores are only comparable within a shot at
//! the same downscaled dimensions.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{LumaFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::noise::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//!
//! # let bytes = vec![0u8; 256 * 144];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let luma = LumaFrame::new(&bytes, 256, 144, 256, Timestamp::new(0, tb));
//! let sigma = det.observe_luma(luma);
//! assert!(sigma >= 0.0);
//! ```

use crate::frame::LumaFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the noise detector.
///
/// Currently only carries the `use_simd` flag for forward-compatibility;
/// the scalar path is always used in v0.1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  use_simd: bool,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
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

/// Pure-algo state machine that reduces a luma frame to its Immerkaer
/// noise estimate σₙ in 0-255 space.
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

  /// Computes the Immerkaer noise estimate σₙ on `luma`. Frames narrower
  /// or shorter than 3 pixels have no interior and yield `0.0`.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32 {
    crate::arch::noise(
      luma.data(),
      luma.width() as usize,
      luma.height() as usize,
      luma.stride() as usize,
      self.opts.use_simd,
    )
  }
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
    Timestamp::new(0, Timebase::new(1, nz(1_000_000)))
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
  fn uniform_frame_has_zero_noise() {
    let data = vec![100u8; 32 * 32];
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, 32, 32));
    assert_eq!(sigma, 0.0);
  }

  #[test]
  fn too_small_frame_yields_zero() {
    let data = vec![0u8; 4];
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, 2, 2));
    assert_eq!(sigma, 0.0);
  }

  #[test]
  fn checkerboard_matches_closed_form() {
    // ±64 amplitude checkerboard → per-pixel |lap| = 1024 → σₙ ≈ 213.92.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        let phase = ((x + y) & 1) as i32;
        let val = 100i32 + if phase == 0 { -64 } else { 64 };
        data[y * w + x] = val as u8;
      }
    }
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    let expected = 0.208_898_754_886_372_3_f64 * 1024.0;
    assert!(
      ((sigma as f64) - expected).abs() < 0.5,
      "expected ~{expected}, got {sigma}"
    );
  }

  #[test]
  fn stride_padding_is_ignored() {
    let w = 4usize;
    let h = 4usize;
    let stride = 8usize;
    let mut data = vec![255u8; stride * h];
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 100;
      }
    }
    let f = LumaFrame::new(&data, w as u32, h as u32, stride as u32, timestamp());
    let mut det = Detector::new(Options::default());
    let sigma = det.observe_luma(f);
    assert_eq!(sigma, 0.0, "padding leaked into kernel");
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![0u8; 16 * 16];
    let sigma = det.observe_luma(tight_luma(&data, 16, 16));
    assert_eq!(sigma, 0.0);
  }
}
