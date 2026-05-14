//! Luma-plane mean and variance detector.
//!
//! Consumes a single-plane 8-bit [`LumaFrame`] and emits the frame's
//! arithmetic mean and population variance in 0-255 space. These two
//! numbers together drive the "black" / "overexposed" / "flat" gates in
//! the selection state machine.
//!
//! The kernel is a straightforward two-pass reduction — cheap on a 256-px
//! downscale (the preprocess layer's target) and numerically stable for
//! values bounded to `[0, 255]`. A SIMD path is reserved via
//! [`Options::with_simd`] but not yet implemented.
//!
//! # Example
//!
//! ```no_run
//! use core::num::NonZeroU32;
//! use scenesdetect::frame::{LumaFrame, Timebase, Timestamp};
//! use scenesdetect::keyframe::luma::{Detector, Options};
//!
//! let mut det = Detector::new(Options::default());
//!
//! # let bytes = vec![128u8; 256 * 144];
//! # let tb = Timebase::new(1, NonZeroU32::new(1_000_000).unwrap());
//! # let luma = LumaFrame::new(&bytes, 256, 144, 256, Timestamp::new(0, tb));
//! let stats = det.observe_luma(luma);
//! assert!((stats.mean() - 128.0).abs() < 1e-3);
//! assert!(stats.variance() < 1e-3); // uniform frame
//! ```

use crate::frame::LumaFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Mean and population variance of a luma plane in 0-255 space.
///
/// Fields are private; use [`mean`](Self::mean) /
/// [`variance`](Self::variance) for reads, [`with_mean`](Self::with_mean)
/// / [`with_variance`](Self::with_variance) for `const fn` builders,
/// and [`set_mean`](Self::set_mean) / [`set_variance`](Self::set_variance)
/// for in-place mutation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LumaStats {
  mean: f32,
  variance: f32,
}

impl Default for LumaStats {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl LumaStats {
  /// Creates an all-zero [`LumaStats`] (same value as
  /// [`LumaStats::default`], usable in `const` contexts).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      mean: 0.0,
      variance: 0.0,
    }
  }

  /// Arithmetic mean of all sampled luma pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mean(&self) -> f32 {
    self.mean
  }
  /// Population variance of the sampled luma pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn variance(&self) -> f32 {
    self.variance
  }

  /// Returns `self` with [`mean`](Self::mean) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_mean(mut self, v: f32) -> Self {
    self.mean = v;
    self
  }
  /// Returns `self` with [`variance`](Self::variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_variance(mut self, v: f32) -> Self {
    self.variance = v;
    self
  }

  /// In-place setter for [`mean`](Self::mean). Returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_mean(&mut self, v: f32) -> &mut Self {
    self.mean = v;
    self
  }
  /// In-place setter for [`variance`](Self::variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_variance(&mut self, v: f32) -> &mut Self {
    self.variance = v;
    self
  }
}

/// Options for the luma detector.
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

/// Pure-algo state machine that reduces a luma frame to [`LumaStats`].
///
/// The "state machine" name is aspirational — the current implementation
/// is stateless beyond its [`Options`]. `clear` is provided for parity
/// with the rest of the crate's detectors and is a no-op today.
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

  /// Resets any stream state. No-op today; reserved for future SIMD
  /// backends that may want per-stream warmup caches.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the mean and variance of `luma`. The caller already has
  /// the frame's timestamp via [`LumaFrame::timestamp`] — it is not
  /// re-emitted here.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> LumaStats {
    luma_stats_scalar(&luma, self.opts.use_simd())
  }
}

/// Thin wrapper over [`super::reduce::plane_mean_variance`] that
/// unpacks a [`LumaFrame`] into its plane-and-dims tuple.
fn luma_stats_scalar(luma: &LumaFrame<'_>, use_simd: bool) -> LumaStats {
  let (mean, variance) = super::reduce::plane_mean_variance(
    luma.data(),
    luma.width() as usize,
    luma.height() as usize,
    luma.stride() as usize,
    use_simd,
  );
  LumaStats::new().with_mean(mean).with_variance(variance)
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
    Timestamp::new(1234, Timebase::new(1, nz(1000)))
  }

  #[test]
  fn options_default_enables_simd() {
    assert!(Options::default().use_simd());
  }

  #[test]
  fn options_builder_roundtrips() {
    let o = Options::new().with_simd(false);
    assert!(!o.use_simd());
    let o = o.with_simd(true);
    assert!(o.use_simd());
  }

  #[test]
  fn black_frame_has_zero_mean_and_variance() {
    let data = vec![0u8; 64 * 48];
    let f = LumaFrame::new(&data, 64, 48, 64, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert_eq!(stats.mean(), 0.0);
    assert_eq!(stats.variance(), 0.0);
  }

  #[test]
  fn uniform_gray_has_zero_variance() {
    let data = vec![128u8; 64 * 48];
    let f = LumaFrame::new(&data, 64, 48, 64, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert!((stats.mean() - 128.0).abs() < 1e-3);
    assert!(stats.variance() < 1e-3);
  }

  #[test]
  fn uniform_white_has_zero_variance() {
    let data = vec![255u8; 32 * 32];
    let f = LumaFrame::new(&data, 32, 32, 32, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert!((stats.mean() - 255.0).abs() < 1e-3);
    assert!(stats.variance() < 1e-3);
  }

  #[test]
  fn half_black_half_white_variance_matches_expected() {
    // Half of pixels at 0, half at 255 → mean = 127.5, var = 127.5² = 16256.25.
    let mut data = vec![0u8; 32 * 32];
    for y in 16..32 {
      for x in 0..32 {
        data[y * 32 + x] = 255;
      }
    }
    let f = LumaFrame::new(&data, 32, 32, 32, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert!((stats.mean() - 127.5).abs() < 1e-3);
    assert!((stats.variance() - 127.5_f32 * 127.5).abs() < 1.0);
  }

  #[test]
  fn stride_padding_is_ignored() {
    // 4×2 pixels at stride 8 (4 bytes of padding per row). Padding is
    // 255 (should be ignored), pixel data is 0.
    let stride = 8usize;
    let h = 2usize;
    let w = 4usize;
    let mut data = vec![255u8; stride * h];
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 0;
      }
    }
    let f = LumaFrame::new(&data, w as u32, h as u32, stride as u32, timestamp());
    let mut det = Detector::new(Options::default());
    let stats = det.observe_luma(f);
    assert_eq!(stats.mean(), 0.0, "padding bytes leaked into mean");
    assert_eq!(stats.variance(), 0.0, "padding bytes leaked into variance");
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    // Still works after clear.
    let data = vec![100u8; 16 * 16];
    let f = LumaFrame::new(&data, 16, 16, 16, timestamp());
    let stats = det.observe_luma(f);
    assert!((stats.mean() - 100.0).abs() < 1e-3);
  }

  #[test]
  fn lumastats_builders_and_setters_roundtrip() {
    let s = LumaStats::new().with_mean(120.0).with_variance(500.0);
    assert_eq!(s.mean(), 120.0);
    assert_eq!(s.variance(), 500.0);

    let mut s = LumaStats::new();
    s.set_mean(7.0).set_variance(8.0);
    assert_eq!(s.mean(), 7.0);
    assert_eq!(s.variance(), 8.0);
  }

  #[test]
  fn lumastats_new_is_const_context_usable() {
    const S: LumaStats = LumaStats::new().with_mean(3.0).with_variance(4.0);
    assert_eq!(S.mean(), 3.0);
    assert_eq!(S.variance(), 4.0);
  }
}
