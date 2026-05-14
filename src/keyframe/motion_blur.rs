//! Gradient-anisotropy motion-blur estimator.
//!
//! Heuristic: a frame with motion blur in direction θ has gradients
//! that concentrate along the orientation perpendicular to θ; a
//! sharp, scene-rich frame has a more uniform distribution of
//! gradient directions. The detector runs a 3×3 Sobel on the luma
//! plane (via the `Sobel` kernel), builds a 4-bin magnitude-
//! weighted histogram of the quantized direction, and reports
//! `max((max_bin / total) - 0.25, 0) / 0.75`. The output is in
//! `[0, 1]` with 0 = isotropic and 1 = concentrated in a single
//! direction.
//!
//! Caveat: the metric also fires on scenes with a single dominant
//! orientation (e.g. forest canopies, building façades). The
//! selection layer treats motion_blur as opt-in for the strict gate
//! and defaults the composite-argmax weight to zero.
//!
//! Two entry points:
//!
//! - [`Detector::observe_luma`]: runs the full pipeline (internal
//!   Sobel scratch buffers grow monotonically).
//! - [`Detector::observe_sobel`]: skips the Sobel stage when the
//!   caller already has magnitude and direction planes.

use crate::frame::LumaFrame;
use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Options for the motion-blur detector.
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

/// Pure-algo state machine that reduces a luma frame to an anisotropy
/// score in `[0, 1]`. Owns scratch buffers for Sobel magnitude and
/// direction (plus a tight-pack scratch used when the input
/// [`LumaFrame`] has `stride > width`); they grow to the largest frame
/// seen.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
  mag: Vec<i32>,
  dir: Vec<u8>,
  /// Scratch used by [`Detector::observe_luma`] only when the input
  /// frame is stride-padded. The underlying [`crate::arch::sobel`]
  /// kernel is not stride-aware (its row indexing is `y * width`), so
  /// padded rows are repacked here before reduction.
  tight: Vec<u8>,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(opts: Options) -> Self {
    Self {
      opts,
      mag: Vec::new(),
      dir: Vec::new(),
      tight: Vec::new(),
    }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Resets stream state. No-op today; reserved for future SIMD caches.
  /// Does not free the Sobel scratch buffers — they stay grown for
  /// reuse across shots.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {}

  /// Computes the anisotropy score on `luma` and returns it in
  /// `[0, 1]`. Runs Sobel internally into owned scratch buffers, then
  /// reduces. Frames narrower or shorter than 3 pixels return `0.0`.
  ///
  /// Stride-padded `LumaFrame` inputs (`stride > width`) are repacked
  /// into an owned tight scratch before reduction, because the
  /// underlying [`crate::arch::sobel`] kernel is not stride-aware.
  /// The repack is a single contiguous-row copy per frame and grows
  /// monotonically with the largest input seen.
  pub fn observe_luma(&mut self, luma: LumaFrame<'_>) -> f32 {
    let w = luma.width() as usize;
    let h = luma.height() as usize;
    if w < 3 || h < 3 {
      return 0.0;
    }
    let n = w.saturating_mul(h);
    if n == 0 {
      return 0.0;
    }
    if self.mag.len() < n {
      self.mag.resize(n, 0);
    }
    if self.dir.len() < n {
      self.dir.resize(n, 0);
    }

    // arch::sobel uses `y * width` row indexing — not stride-aware.
    // For tight frames (the common case from the preprocess pipeline)
    // we borrow the buffer directly; for stride-padded frames we
    // repack into the owned scratch first so Sobel sees a tight w×h
    // plane.
    let stride = luma.stride() as usize;
    let src: &[u8] = if stride == w {
      &luma.data()[..n]
    } else {
      if self.tight.len() < n {
        self.tight.resize(n, 0);
      }
      let data = luma.data();
      for y in 0..h {
        let soff = y * stride;
        let doff = y * w;
        self.tight[doff..doff + w].copy_from_slice(&data[soff..soff + w]);
      }
      &self.tight[..n]
    };

    crate::arch::sobel(
      src,
      &mut self.mag[..n],
      &mut self.dir[..n],
      w,
      h,
      self.opts.use_simd,
    );
    crate::arch::gradient_anisotropy(&self.mag[..n], &self.dir[..n], w, h, self.opts.use_simd)
  }

  /// Skips the Sobel stage of [`Self::observe_luma`] — the caller
  /// already has the magnitude and direction planes. `mag` and `dir`
  /// are tight-packed `width × height`. On equivalent inputs, this
  /// returns the same value as [`Self::observe_luma`].
  pub fn observe_sobel(&mut self, mag: &[i32], dir: &[u8], width: usize, height: usize) -> f32 {
    crate::arch::gradient_anisotropy(mag, dir, width, height, self.opts.use_simd)
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
  fn uniform_frame_yields_zero_anisotropy() {
    let data = vec![100u8; 32 * 32];
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, 32, 32));
    assert_eq!(a, 0.0);
  }

  #[test]
  fn vertical_edge_is_highly_anisotropic() {
    // Strong vertical edge → gradient direction concentrates in one
    // bin → anisotropy → 1.0.
    let (w, h) = (32usize, 32usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in (w / 2)..w {
        data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, w as u32, h as u32));
    assert!(a > 0.9, "expected strong anisotropy (>0.9), got {a}");
  }

  #[test]
  fn too_small_frame_yields_zero() {
    let data = vec![0u8; 4];
    let mut det = Detector::new(Options::default());
    let a = det.observe_luma(tight_luma(&data, 2, 2));
    assert_eq!(a, 0.0);
  }

  #[test]
  fn stride_padded_input_matches_tight() {
    // Same pixel content fed twice: once as a tight `w×h` frame, once
    // as a `w×h` frame inside a wider `stride×h` buffer with poisonous
    // 0xAA padding bytes. observe_luma must repack and produce
    // identical anisotropy.
    let (w, h) = (16usize, 16usize);
    let mut tight = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        tight[y * w + x] = 255;
      }
    }
    let mut det_tight = Detector::new(Options::default());
    let from_tight = det_tight.observe_luma(tight_luma(&tight, w as u32, h as u32));

    let stride = 32usize;
    let mut padded = vec![0xAAu8; stride * h];
    for y in 0..h {
      padded[y * stride..y * stride + w].copy_from_slice(&tight[y * w..(y + 1) * w]);
    }
    let mut det_padded = Detector::new(Options::default());
    let padded_frame = LumaFrame::new(&padded, w as u32, h as u32, stride as u32, timestamp());
    let from_padded = det_padded.observe_luma(padded_frame);

    assert!(
      (from_tight - from_padded).abs() < 1e-6,
      "padded path {from_padded} ≠ tight path {from_tight}"
    );
    assert!(
      from_padded > 0.9,
      "vertical edge should be strongly anisotropic; got {from_padded}"
    );
  }

  #[test]
  fn observe_sobel_matches_observe_luma() {
    // Build a frame, run observe_luma, then build mag/dir externally
    // and confirm observe_sobel produces the same result.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 8..w {
        data[y * w + x] = 255;
      }
    }
    let mut det = Detector::new(Options::default());
    let from_luma = det.observe_luma(tight_luma(&data, w as u32, h as u32));

    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    crate::arch::sobel(&data, &mut mag, &mut dir, w, h, false);
    let from_sobel = det.observe_sobel(&mag, &dir, w, h);

    assert!(
      (from_luma - from_sobel).abs() < 1e-6,
      "luma path {from_luma} != sobel path {from_sobel}"
    );
  }

  #[test]
  fn clear_is_noop() {
    let mut det = Detector::new(Options::default());
    det.clear();
    let data = vec![0u8; 16 * 16];
    let a = det.observe_luma(tight_luma(&data, 16, 16));
    assert_eq!(a, 0.0);
  }
}
