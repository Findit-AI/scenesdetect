//! Perceptual hash (pHash) scene detection via DCT signatures.
//!
//! This module implements [`Detector`](crate::phash::Detector), a port of
//! PySceneDetect's `detect-hash` algorithm. Where
//! [`histogram::Detector`](crate::histogram::Detector) looks at *brightness
//! distribution*, the pHash detector looks at *spatial structure*: a cut
//! fires when the low-frequency DCT signature of the frame changes
//! significantly.
//!
//! # Algorithm
//!
//! For each incoming [`LumaFrame`](crate::frame::LumaFrame):
//!
//! 1. **Resize** the Y plane to `imsize × imsize` (where `imsize = size *
//!    lowpass`) using area-weighted downsampling.
//! 2. **Normalize** to `[0, 1]` by dividing by the max sample.
//! 3. **2D DCT-II** (orthonormal, matching OpenCV's `cv2.dct` scaling) on
//!    the resized image.
//! 4. **Crop** to the top-left `size × size` low-frequency block.
//! 5. **Median threshold:** set bit `i` iff that coefficient is strictly
//!    greater than the block's median.
//!
//! The resulting `size²` bits are the frame's pHash. Between consecutive
//! frames, the normalized Hamming distance
//! `popcount(h1 ^ h2) / (size²)` is compared against `threshold`; a cut is
//! emitted when it is `>=` and at least `min_duration` has elapsed since the
//! previous cut.
//!
//! Default parameters (`size=16`, `lowpass=2`) → resize to `32 × 32`, DCT,
//! then a `16 × 16 = 256`-bit fingerprint per frame. Comparison cost is a
//! handful of `XOR` + `popcount` instructions.
//!
//! # Attribution
//!
//! Based on Neal Krawetz's DCT-based pHash (2011) and Johannes Buchner's
//! `imagehash` library. Directly ported from PySceneDetect's `detect-hash`
//! (BSD 3-Clause).

use core::{f32::consts::PI, time::Duration};
use derive_more::IsVariant;
use thiserror::Error;

use crate::frame::{LumaFrame, Timebase, Timestamp};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{vec, vec::Vec};

use super::{ceil_32, cos_32, floor_32, sqrt_32};

/// Configuration for [`Detector`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  threshold: f64,
  size: u32,
  lowpass: u32,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  min_duration: Duration,
  allow_initial_cut: bool,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Creates a new [`Options`] with the specified parameters.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold: 0.395,
      size: 16,
      lowpass: 2,
      min_duration: Duration::from_secs(1),
      allow_initial_cut: true,
    }
  }

  /// Returns the threshold for scene change detection. Higher values are more sensitive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }

  /// Sets the scene change threshold. Higher values are more sensitive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold(mut self, threshold: f64) -> Self {
    self.set_threshold(threshold);
    self
  }

  /// Sets the scene change threshold. Higher values are more sensitive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threshold(&mut self, threshold: f64) -> &mut Self {
    self.threshold = threshold;
    self
  }

  /// Returns the hash size. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn size(&self) -> u32 {
    self.size
  }

  /// Sets the hash size. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_size(mut self, size: u32) -> Self {
    self.set_size(size);
    self
  }

  /// Sets the hash size. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_size(&mut self, size: u32) -> &mut Self {
    self.size = size;
    self
  }

  /// Returns the lowpass filter size used to smooth the image before hashing. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn lowpass(&self) -> u32 {
    self.lowpass
  }

  /// Sets the lowpass filter size. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_lowpass(mut self, lowpass: u32) -> Self {
    self.set_lowpass(lowpass);
    self
  }

  /// Sets the lowpass filter size. Higher values are more sensitive but more expensive.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_lowpass(&mut self, lowpass: u32) -> &mut Self {
    self.lowpass = lowpass;
    self
  }

  /// Returns the minimum scene duration. Shorter scenes are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_duration(&self) -> Duration {
    self.min_duration
  }

  /// Sets the minimum scene duration. Shorter scenes are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_duration(mut self, min_duration: Duration) -> Self {
    self.set_min_duration(min_duration);
    self
  }

  /// Sets the minimum scene duration. Shorter scenes are ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_duration(&mut self, min_duration: Duration) -> &mut Self {
    self.min_duration = min_duration;
    self
  }

  /// Set the minimum scene length as a number of frames at a given frame rate.
  ///
  /// Convenience for users coming from frame-count APIs (e.g., PySceneDetect's
  /// `min_scene_len`). Internally this converts to [`Self::min_duration`] via
  /// [`Timebase::frames_to_duration`]. On VFR content the duration stays fixed
  /// while frame counts drift — that's the desired behavior.
  ///
  /// `fps` is interpreted as frames per second: 30 fps = `Timebase::new(30, 1)`,
  /// NTSC = `Timebase::new(30000, 1001)`.
  ///
  /// # Panics
  ///
  /// Panics if `fps.num() == 0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_frames(mut self, frames: u32, fps: Timebase) -> Self {
    self.set_min_frames(frames, fps);
    self
  }

  /// In-place form of [`Self::with_min_frames`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_frames(&mut self, frames: u32, fps: Timebase) -> &mut Self {
    self.min_duration = fps.frames_to_duration(frames);
    self
  }

  /// Whether the first detected cut is allowed to fire immediately.
  ///
  /// - `true` (default): the first detected cut fires as soon as the
  ///   normalized Hamming distance exceeds `threshold`.
  /// - `false`: suppresses cuts until the stream has actually run for at
  ///   least [`Self::min_duration`]. Matches PySceneDetect's default.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn allow_initial_cut(&self) -> bool {
    self.allow_initial_cut
  }

  /// Sets whether the first detected cut may fire immediately.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_allow_initial_cut(mut self, val: bool) -> Self {
    self.allow_initial_cut = val;
    self
  }

  /// Sets `allow_initial_cut` in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_allow_initial_cut(&mut self, val: bool) -> &mut Self {
    self.allow_initial_cut = val;
    self
  }
}

/// Error returned by [`Detector::try_new`] when the provided [`Options`] are
/// inconsistent.
#[derive(Debug, Clone, PartialEq, Eq, IsVariant, Error)]
#[non_exhaustive]
pub enum Error {
  /// `options.size() < 2`. The algorithm needs at least a `2 × 2` hash block
  /// to have a meaningful median threshold.
  #[error("phash size ({size}) must be >= 2")]
  SizeTooSmall {
    /// The provided size.
    size: u32,
  },
  /// `options.lowpass() < 1`. The resize multiplier must be at least 1 so
  /// that `imsize = size * lowpass >= size`.
  #[error("phash lowpass ({lowpass}) must be >= 1")]
  LowpassTooSmall {
    /// The provided lowpass multiplier.
    lowpass: u32,
  },
  /// `size * lowpass` or its square would exceed `usize`. Only reachable
  /// with pathological values on 32-bit targets.
  #[error("phash dimensions overflow usize: size ({size}) * lowpass ({lowpass}) squared")]
  DimensionsOverflow {
    /// The provided size.
    size: u32,
    /// The provided lowpass multiplier.
    lowpass: u32,
  },
}

/// Perceptual-hash scene detector. See the
/// [module-level documentation](crate::phash) for the algorithm.
///
/// After construction the detector allocates nothing per frame: the DCT
/// cosine basis matrix is precomputed, and scratch buffers for the resized
/// image, the DCT intermediate/result, the low-frequency block, and a sort
/// scratch for the median are all reused.
#[derive(Debug, Clone)]
pub struct Detector {
  options: Options,
  /// `size * lowpass` — side length of the resized square image.
  imsize: usize,
  /// `options.size` as `usize` — side length of the low-frequency block.
  size: usize,
  /// `options.threshold` cached as f64 for fast comparison.
  threshold: f64,
  /// Precomputed orthonormal DCT-II basis: `dct_cos[k*imsize + n] = α(k) · cos(π(2n+1)k / 2N)`.
  dct_cos: Vec<f32>,
  /// Area-weighted resize weights. Lazily built on the first frame, then
  /// reused across frames of matching dimensions. Rebuilt if the input
  /// resolution changes mid-stream (seeks, adaptive bitrate).
  resize_table: ResizeTable,
  /// Resized (`imsize × imsize`) and normalized (`[0, 1]`) image.
  resized: Vec<f32>,
  /// Row-transformed intermediate for the 2D DCT.
  dct_tmp: Vec<f32>,
  /// Full 2D DCT result.
  dct_result: Vec<f32>,
  /// Flattened `size × size` low-frequency crop (order preserved for bit packing).
  low_freq: Vec<f32>,
  /// Sort scratch for the median — avoids disturbing `low_freq`.
  sort_scratch: Vec<f32>,
  /// Packed bits of the current frame's hash; `len = ceil(size² / 64)`.
  current_hash: Vec<u64>,
  /// Packed bits of the previous frame's hash.
  previous_hash: Vec<u64>,
  has_previous: bool,
  last_cut_ts: Option<Timestamp>,
  last_distance: Option<f64>,
}

impl Detector {
  /// Creates a new detector with the given options, validating them.
  ///
  /// Prefer [`Self::try_new`] at runtime call sites where invalid options
  /// are possible; this constructor is meant for call sites where the
  /// options are statically known-good (tests, fixtures, defaults).
  ///
  /// # Panics
  ///
  /// Panics if the options are invalid — see [`enum@Error`] for the specific
  /// conditions.
  pub fn new(options: Options) -> Self {
    Self::try_new(options).expect("invalid phash Options")
  }

  /// Creates a new detector with the given options, returning [`enum@Error`] if
  /// the options are inconsistent.
  ///
  /// Validates:
  /// - `options.size() >= 2` (need a non-trivial hash block)
  /// - `options.lowpass() >= 1` (need at least unit resize)
  /// - `size * lowpass * size * lowpass` fits in `usize` (avoids overflow
  ///   when sizing scratch buffers on 32-bit targets)
  ///
  /// Precomputes the DCT basis and allocates all scratch buffers on success.
  pub fn try_new(options: Options) -> Result<Self, Error> {
    if options.size < 2 {
      return Err(Error::SizeTooSmall { size: options.size });
    }
    if options.lowpass < 1 {
      return Err(Error::LowpassTooSmall {
        lowpass: options.lowpass,
      });
    }

    let size = options.size as usize;
    let lowpass = options.lowpass as usize;
    let imsize = match size.checked_mul(lowpass) {
      Some(v) => v,
      None => {
        return Err(Error::DimensionsOverflow {
          size: options.size,
          lowpass: options.lowpass,
        });
      }
    };
    let total = match imsize.checked_mul(imsize) {
      Some(v) => v,
      None => {
        return Err(Error::DimensionsOverflow {
          size: options.size,
          lowpass: options.lowpass,
        });
      }
    };

    let threshold = options.threshold;
    let bits = size * size;
    let hash_words = bits.div_ceil(64);
    let dct_cos = build_dct_cos(imsize);

    Ok(Self {
      options,
      imsize,
      size,
      threshold,
      dct_cos,
      resize_table: ResizeTable::new(),
      resized: vec![0.0f32; total],
      dct_tmp: vec![0.0f32; total],
      dct_result: vec![0.0f32; total],
      low_freq: vec![0.0f32; bits],
      sort_scratch: vec![0.0f32; bits],
      current_hash: vec![0u64; hash_words],
      previous_hash: vec![0u64; hash_words],
      has_previous: false,
      last_cut_ts: None,
      last_distance: None,
    })
  }

  /// Returns a reference to the options used by this detector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// Returns the normalized Hamming distance between the last two frames'
  /// hashes, or `None` if fewer than two frames have been processed.
  ///
  /// Range: `[0.0, 1.0]`. `0.0` means identical hashes; `1.0` means every
  /// bit flipped. Useful for logging / diagnostics.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_distance(&self) -> Option<f64> {
    self.last_distance
  }

  /// Resets the detector's streaming state so it can be reused on a fresh
  /// stream (e.g., when the next video begins) without rebuilding the DCT
  /// basis or reallocating scratch buffers.
  ///
  /// After `clear()` the next [`Self::process`] call is treated as if it
  /// were the first frame of a new stream: no cut is emitted, and the frame
  /// re-seeds `last_cut_ts`. The previous video's hashes, `last_cut_ts`,
  /// and `last_distance` are all discarded.
  ///
  /// The resize table is kept. It will reuse its weights if the new stream
  /// has the same resolution, or auto-rebuild on the first frame otherwise.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.has_previous = false;
    self.last_cut_ts = None;
    self.last_distance = None;
  }

  /// Processes the next frame. Returns `Some(ts)` if a cut is detected at
  /// the frame's timestamp, otherwise `None`.
  ///
  /// The first frame establishes the baseline hash and cut-gating reference;
  /// no cut is emitted for it.
  pub fn process(&mut self, frame: LumaFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();

    if self.last_cut_ts.is_none() {
      self.last_cut_ts = Some(if self.options.allow_initial_cut {
        ts.saturating_sub_duration(self.options.min_duration)
      } else {
        ts
      });
    }

    self.compute_hash(&frame);

    let mut cut: Option<Timestamp> = None;
    if self.has_previous {
      let dist = hamming_distance(&self.previous_hash, &self.current_hash);
      let bits = self.size * self.size;
      let norm = dist as f64 / bits as f64;
      self.last_distance = Some(norm);

      let min_elapsed = self
        .last_cut_ts
        .as_ref()
        .and_then(|last| ts.duration_since(last))
        .is_some_and(|d| d >= self.options.min_duration);

      if norm >= self.threshold && min_elapsed {
        cut = Some(ts);
        self.last_cut_ts = Some(ts);
      }
    }

    core::mem::swap(&mut self.current_hash, &mut self.previous_hash);
    self.has_previous = true;
    cut
  }

  /// Builds the current frame's hash into `self.current_hash`.
  fn compute_hash(&mut self, frame: &LumaFrame<'_>) {
    // 1. Ensure resize table matches the frame dimensions. This rebuilds on
    //    the first frame and on any subsequent dimension change. For a CFR
    //    stream this cost is paid once.
    self
      .resize_table
      .ensure(frame.width(), frame.height(), self.imsize);

    // 2. Area-weighted downsample, returning `max` in the same pass so we
    //    fold the normalization pre-scan into the resize loop.
    let max = self.resize_table.apply(
      &mut self.resized,
      frame.data(),
      frame.stride() as usize,
      self.imsize,
    );

    // 3. Normalize by max. Second pass over the 1 KiB `resized` buffer.
    let scale = if max == 0.0 { 1.0 } else { 1.0 / max };
    for v in self.resized.iter_mut() {
      *v *= scale;
    }

    // 4. 2D DCT-II (orthonormal, matching cv2.dct).
    dct2(
      &self.dct_cos,
      &self.resized,
      &mut self.dct_tmp,
      &mut self.dct_result,
      self.imsize,
    );

    // 5. Crop top-left size×size block into a flat buffer.
    for y in 0..self.size {
      let src_row = &self.dct_result[y * self.imsize..y * self.imsize + self.size];
      let dst_row = &mut self.low_freq[y * self.size..(y + 1) * self.size];
      dst_row.copy_from_slice(src_row);
    }

    // 6. Median via O(N) quick-select on sort_scratch (preserves `low_freq`).
    self.sort_scratch.clone_from(&self.low_freq);
    let median = median_f32(&mut self.sort_scratch);

    // 7. Pack bits: bit i set iff low_freq[i] > median. Bit 0 = (0,0) = DC term.
    self.current_hash.fill(0);
    for (i, &v) in self.low_freq.iter().enumerate() {
      if v > median {
        self.current_hash[i / 64] |= 1u64 << (i % 64);
      }
    }
  }
}

/// Builds the orthonormal DCT-II basis: `C[k, n] = α(k) · cos(π(2n+1)k / 2N)`,
/// where `α(0) = 1/√N` and `α(k≠0) = √(2/N)`. This matches `cv2.dct`.
fn build_dct_cos(n: usize) -> Vec<f32> {
  let mut c = vec![0.0f32; n * n];
  let alpha0 = sqrt_32(1.0 / n as f32);
  let alpha_k = sqrt_32(2.0 / n as f32);
  for k in 0..n {
    let a = if k == 0 { alpha0 } else { alpha_k };
    for m in 0..n {
      let angle = PI * (2.0 * m as f32 + 1.0) * k as f32 / (2.0 * n as f32);
      c[k * n + m] = a * cos_32(angle);
    }
  }
  c
}

/// Separable 2D DCT-II: `result = C · input · Cᵀ`.
/// `tmp` is a scratch buffer of size `n*n`.
fn dct2(c: &[f32], input: &[f32], tmp: &mut [f32], result: &mut [f32], n: usize) {
  debug_assert_eq!(c.len(), n * n);
  debug_assert_eq!(input.len(), n * n);
  debug_assert_eq!(tmp.len(), n * n);
  debug_assert_eq!(result.len(), n * n);

  // tmp = input · Cᵀ   (row transform; output column j = Σ_k input[m, k] · C[j, k])
  for m in 0..n {
    for j in 0..n {
      let mut s = 0.0f32;
      for k in 0..n {
        s += input[m * n + k] * c[j * n + k];
      }
      tmp[m * n + j] = s;
    }
  }
  // result = C · tmp    (column transform; output[k, j] = Σ_m C[k, m] · tmp[m, j])
  for k in 0..n {
    for j in 0..n {
      let mut s = 0.0f32;
      for m in 0..n {
        s += c[k * n + m] * tmp[m * n + j];
      }
      result[k * n + j] = s;
    }
  }
}

/// Precomputed area-weighted resize weights for a fixed
/// `src_{w,h} → dst_size × dst_size` mapping.
///
/// Factors the 2D area weight as a product of 1D horizontal and vertical
/// overlap fractions. For each destination row / column, we store a
/// contiguous run of `(src_idx, weight)` pairs, indexed via prefix-sum
/// `x_range_starts` / `y_range_starts`. Empty `(src_w = 0, src_h = 0)`
/// is the "not yet built" sentinel — [`Self::ensure`] detects it.
#[derive(Debug, Clone)]
struct ResizeTable {
  src_w: u32,
  src_h: u32,
  inv_area: f32,
  /// Source column indices contributing to each destination column, flattened.
  x_offsets: Vec<u32>,
  x_weights: Vec<f32>,
  /// Prefix sum; `x_range_starts[dst_x]..x_range_starts[dst_x+1]` indexes
  /// the contiguous run of pairs for destination column `dst_x`. Length
  /// `dst_size + 1`.
  x_range_starts: Vec<u32>,
  /// Same, for rows.
  y_offsets: Vec<u32>,
  y_weights: Vec<f32>,
  y_range_starts: Vec<u32>,
}

impl ResizeTable {
  /// Creates an empty (not-yet-built) table.
  fn new() -> Self {
    Self {
      src_w: 0,
      src_h: 0,
      inv_area: 0.0,
      x_offsets: Vec::new(),
      x_weights: Vec::new(),
      x_range_starts: Vec::new(),
      y_offsets: Vec::new(),
      y_weights: Vec::new(),
      y_range_starts: Vec::new(),
    }
  }

  /// Ensures the table matches the given dimensions, rebuilding if needed.
  ///
  /// Fast path when dimensions are unchanged: single comparison, no work.
  fn ensure(&mut self, src_w: u32, src_h: u32, dst_size: usize) {
    if self.src_w == src_w && self.src_h == src_h {
      return;
    }
    self.rebuild(src_w, src_h, dst_size);
  }

  /// Rebuilds the table for the given dimensions. Reuses existing `Vec`
  /// capacity via `clear` — no heap churn after the first resolution.
  fn rebuild(&mut self, src_w: u32, src_h: u32, dst_size: usize) {
    debug_assert!(src_w > 0 && src_h > 0, "source dimensions must be non-zero");
    debug_assert!(dst_size > 0);

    self.x_offsets.clear();
    self.x_weights.clear();
    self.x_range_starts.clear();
    self.y_offsets.clear();
    self.y_weights.clear();
    self.y_range_starts.clear();

    let scale_x = src_w as f32 / dst_size as f32;
    let scale_y = src_h as f32 / dst_size as f32;

    build_axis(
      &mut self.x_offsets,
      &mut self.x_weights,
      &mut self.x_range_starts,
      src_w,
      dst_size,
      scale_x,
    );
    build_axis(
      &mut self.y_offsets,
      &mut self.y_weights,
      &mut self.y_range_starts,
      src_h,
      dst_size,
      scale_y,
    );

    self.inv_area = 1.0 / (scale_x * scale_y);
    self.src_w = src_w;
    self.src_h = src_h;
  }

  /// Applies the table to an 8-bit source plane, writing f32 values into
  /// `dst` and returning the max value seen — so the normalization pre-scan
  /// is folded into this single pass.
  fn apply(&self, dst: &mut [f32], src: &[u8], src_stride: usize, dst_size: usize) -> f32 {
    debug_assert_eq!(dst.len(), dst_size * dst_size);
    debug_assert_eq!(self.x_range_starts.len(), dst_size + 1);
    debug_assert_eq!(self.y_range_starts.len(), dst_size + 1);

    let mut max = 0.0f32;

    for dst_y in 0..dst_size {
      let y_start = self.y_range_starts[dst_y] as usize;
      let y_end = self.y_range_starts[dst_y + 1] as usize;

      for dst_x in 0..dst_size {
        let x_start = self.x_range_starts[dst_x] as usize;
        let x_end = self.x_range_starts[dst_x + 1] as usize;

        let mut sum = 0.0f32;
        for yi in y_start..y_end {
          let sy = self.y_offsets[yi] as usize;
          let wy = self.y_weights[yi];
          let row_off = sy * src_stride;
          let mut row_sum = 0.0f32;
          for xi in x_start..x_end {
            let sx = self.x_offsets[xi] as usize;
            row_sum += (src[row_off + sx] as f32) * self.x_weights[xi];
          }
          sum += row_sum * wy;
        }

        let v = sum * self.inv_area;
        dst[dst_y * dst_size + dst_x] = v;
        if v > max {
          max = v;
        }
      }
    }

    max
  }
}

/// Populates one axis (horizontal or vertical) of a resize table. Pushes
/// `(src_idx, weight)` pairs to `offsets`/`weights` and `range_starts`
/// entries such that `range_starts[dst]..range_starts[dst+1]` is the run of
/// pairs for destination index `dst`. The final `range_starts.len()` is
/// `dst_size + 1` (prefix-sum style — last entry is the total length).
fn build_axis(
  offsets: &mut Vec<u32>,
  weights: &mut Vec<f32>,
  range_starts: &mut Vec<u32>,
  src_size: u32,
  dst_size: usize,
  scale: f32,
) {
  for dst in 0..dst_size {
    range_starts.push(offsets.len() as u32);
    let a = dst as f32 * scale;
    let b = (dst + 1) as f32 * scale;
    let s_start = floor_32(a) as u32;
    let s_end = (ceil_32(b) as u32).min(src_size);
    for s in s_start..s_end {
      let w = ((s + 1) as f32).min(b) - (s as f32).max(a);
      if w > 0.0 {
        offsets.push(s);
        weights.push(w);
      }
    }
  }
  range_starts.push(offsets.len() as u32);
}

/// Median of a slice in O(N) via quick-select. Destroys the input order.
///
/// For odd `n`, returns the (`n/2`)th order statistic directly. For even
/// `n`, returns the average of the (`n/2 − 1`)th and (`n/2`)th — matching
/// `numpy.median` and therefore PySceneDetect.
fn median_f32(buf: &mut [f32]) -> f32 {
  let n = buf.len();
  debug_assert!(n > 0);
  if n == 1 {
    return buf[0];
  }
  let mid = n / 2;
  let (left, pivot, _right) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
  let m2 = *pivot;
  if n % 2 == 1 {
    m2
  } else {
    // Even length: also need the (mid − 1)th order statistic, which is the
    // max of the left partition produced by the select above.
    let m1 = left.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (m1 + m2) / 2.0
  }
}

/// Hamming distance between two equal-length bit strings stored as `u64` words.
#[cfg_attr(not(tarpaulin), inline(always))]
fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
  debug_assert_eq!(a.len(), b.len());
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| (x ^ y).count_ones())
    .sum()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frame::Timebase;
  use core::num::NonZeroU32;
  use std::{vec, vec::Vec};

  const fn nz32(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
      Some(v) => v,
      None => panic!("zero"),
    }
  }

  fn make_frame<'a>(data: &'a [u8], w: u32, h: u32, pts: i64) -> LumaFrame<'a> {
    let tb = Timebase::new(1, nz32(1000));
    LumaFrame::new(data, w, h, w, Timestamp::new(pts, tb))
  }

  #[test]
  fn with_min_frames_matches_python_default() {
    // PySceneDetect's default is 15 frames; at 30 fps that's 500 ms.
    let fps = Timebase::new(30, nz32(1));
    let opts = Options::default().with_min_frames(15, fps);
    assert_eq!(opts.min_duration(), Duration::from_millis(500));
  }

  #[test]
  fn with_min_frames_ntsc() {
    let fps = Timebase::new(30_000, nz32(1001));
    let opts = Options::default().with_min_frames(15, fps);
    assert_eq!(opts.min_duration(), Duration::from_nanos(500_500_000));
  }

  #[test]
  fn try_new_success() {
    let det = Detector::try_new(Options::default()).expect("defaults are valid");
    assert_eq!(det.options().size(), 16);
    assert_eq!(det.options().lowpass(), 2);
  }

  #[test]
  fn try_new_rejects_size_too_small() {
    let opts = Options::default().with_size(1);
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::SizeTooSmall { size: 1 });

    let opts = Options::default().with_size(0);
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::SizeTooSmall { size: 0 });
  }

  #[test]
  fn try_new_rejects_lowpass_zero() {
    let opts = Options::default().with_lowpass(0);
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::LowpassTooSmall { lowpass: 0 });
  }

  #[test]
  #[should_panic(expected = "invalid phash Options")]
  fn new_panics_on_invalid() {
    let _ = Detector::new(Options::default().with_size(1));
  }

  #[test]
  fn error_display() {
    let e = Error::SizeTooSmall { size: 1 };
    assert_eq!(format!("{e}"), "phash size (1) must be >= 2");
    let e = Error::LowpassTooSmall { lowpass: 0 };
    assert_eq!(format!("{e}"), "phash lowpass (0) must be >= 1");
  }

  #[test]
  fn hamming_distance_basic() {
    assert_eq!(hamming_distance(&[0, 0], &[0, 0]), 0);
    assert_eq!(hamming_distance(&[0xFF, 0], &[0, 0]), 8);
    assert_eq!(hamming_distance(&[!0u64, !0u64], &[0, 0]), 128);
    assert_eq!(hamming_distance(&[0b1010_1010], &[0b0101_0101]), 8);
  }

  #[test]
  fn build_dct_cos_is_orthonormal() {
    // C · Cᵀ should be the identity for the orthonormal DCT basis.
    let n = 8;
    let c = build_dct_cos(n);
    for i in 0..n {
      for j in 0..n {
        let mut s = 0.0f32;
        for k in 0..n {
          s += c[i * n + k] * c[j * n + k];
        }
        let expected = if i == j { 1.0 } else { 0.0 };
        assert!(
          (s - expected).abs() < 1e-5,
          "C·Cᵀ at ({i},{j}) = {s}, want {expected}",
        );
      }
    }
  }

  #[test]
  fn dct_dc_of_constant_input() {
    // DCT of a constant signal: all energy in the DC bin (0, 0).
    let n = 8;
    let c = build_dct_cos(n);
    let input = vec![1.0f32; n * n];
    let mut tmp = vec![0.0f32; n * n];
    let mut result = vec![0.0f32; n * n];
    dct2(&c, &input, &mut tmp, &mut result, n);
    // DC = α(0)² · n · n · 1 = (1/√n)² · n · n = n  (for each dim)
    // 2D DC = n · α(0)² · n = n for 1D, squared for 2D = n
    // Actually: for orthonormal 2D DCT of constant 1: Y[0,0] = n (since α(0) = 1/√n
    // and summing n values gives n/√n = √n per dim, then 2D = n).
    assert!((result[0] - n as f32).abs() < 1e-4, "DC = {}", result[0]);
    // All other coefficients ≈ 0.
    (1..n * n).for_each(|k| {
      assert!(result[k].abs() < 1e-4, "AC [{k}] = {}", result[k]);
    });
  }

  #[test]
  fn resize_area_identity() {
    // 4x4 → 4x4 is a no-op.
    let src = [
      10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let mut dst = vec![0.0f32; 16];
    let mut table = ResizeTable::new();
    table.ensure(4, 4, 4);
    let max = table.apply(&mut dst, &src, 4, 4);
    for i in 0..16 {
      assert!((dst[i] - src[i] as f32).abs() < 1e-5);
    }
    assert!((max - 160.0).abs() < 1e-5);
  }

  #[test]
  fn resize_area_halve() {
    // 4x4 → 2x2 with a known input — each dest pixel is the average of a 2x2 source block.
    let src = [
      10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let mut dst = vec![0.0f32; 4];
    let mut table = ResizeTable::new();
    table.ensure(4, 4, 2);
    let max = table.apply(&mut dst, &src, 4, 2);
    assert!((dst[0] - (10.0 + 20.0 + 50.0 + 60.0) / 4.0).abs() < 1e-4);
    assert!((dst[1] - (30.0 + 40.0 + 70.0 + 80.0) / 4.0).abs() < 1e-4);
    assert!((dst[2] - (90.0 + 100.0 + 130.0 + 140.0) / 4.0).abs() < 1e-4);
    assert!((dst[3] - (110.0 + 120.0 + 150.0 + 160.0) / 4.0).abs() < 1e-4);
    // apply() returns the max — equals the largest destination pixel.
    assert!((max - 135.0).abs() < 1e-4);
  }

  #[test]
  fn resize_table_rebuild_on_dim_change() {
    let mut table = ResizeTable::new();
    // First build.
    table.ensure(1920, 1080, 32);
    let counts_first = (table.x_offsets.len(), table.y_offsets.len());
    // Same dims — fast no-op.
    table.ensure(1920, 1080, 32);
    assert_eq!(table.x_offsets.len(), counts_first.0);
    // Changed dims — rebuild. Weight counts differ for different src size.
    table.ensure(1280, 720, 32);
    assert_ne!(table.x_offsets.len(), counts_first.0);
    assert_eq!(table.src_w, 1280);
    assert_eq!(table.src_h, 720);
  }

  #[test]
  fn median_odd_and_even() {
    // Odd length: returns the middle element.
    let mut v = [5.0f32, 1.0, 3.0, 2.0, 4.0];
    assert_eq!(median_f32(&mut v), 3.0);
    // Even length: returns average of the two middle elements.
    let mut v = [5.0f32, 1.0, 3.0, 2.0, 4.0, 6.0];
    assert_eq!(median_f32(&mut v), (3.0 + 4.0) / 2.0);
  }

  #[test]
  fn identical_frames_produce_no_cut() {
    let mut det = Detector::new(Options::default());
    // A frame with spatial variation (not flat — we want a meaningful DCT).
    let mut buf = vec![0u8; 128 * 96];
    for (i, b) in buf.iter_mut().enumerate() {
      *b = ((i * 7) % 256) as u8;
    }
    assert!(det.process(make_frame(&buf, 128, 96, 0)).is_none());
    assert!(det.process(make_frame(&buf, 128, 96, 2000)).is_none());
    assert!(det.process(make_frame(&buf, 128, 96, 4000)).is_none());
    assert_eq!(det.last_distance(), Some(0.0));
  }

  /// Returns (top/bottom-half, left/right-half) test frames — orthogonal
  /// low-frequency structures that land clearly inside the 16×16 low-freq
  /// DCT block, so the hashes differ reliably.
  fn ortho_halves_frames() -> (Vec<u8>, Vec<u8>) {
    let mut top_bottom = vec![0u8; 128 * 96];
    for y in 0..96 {
      for x in 0..128 {
        top_bottom[y * 128 + x] = if y < 48 { 220 } else { 30 };
      }
    }
    let mut left_right = vec![0u8; 128 * 96];
    for y in 0..96 {
      for x in 0..128 {
        left_right[y * 128 + x] = if x < 64 { 220 } else { 30 };
      }
    }
    (top_bottom, left_right)
  }

  #[test]
  fn very_different_frames_produce_cut() {
    // Use min_duration=0 so the gate can't mask the cut.
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let (a, b) = ortho_halves_frames();

    assert!(det.process(make_frame(&a, 128, 96, 0)).is_none());
    let cut = det.process(make_frame(&b, 128, 96, 33));
    assert!(
      cut.is_some(),
      "expected cut between top/bottom and left/right halves"
    );
    assert!(
      det.last_distance().unwrap() >= Options::default().threshold(),
      "distance {} should meet default threshold 0.395",
      det.last_distance().unwrap(),
    );
  }

  #[test]
  fn min_duration_suppresses_rapid_cuts() {
    // Python-compat mode: no early cuts allowed.
    let opts = Options::default()
      .with_min_duration(Duration::from_secs(1))
      .with_allow_initial_cut(false);
    let mut det = Detector::new(opts);

    let (a, b) = ortho_halves_frames();

    let mut cuts = 0u32;
    for i in 0..30i64 {
      let frame_data = if i % 2 == 0 { &a } else { &b };
      let ts = i * 33;
      if det.process(make_frame(frame_data, 128, 96, ts)).is_some() {
        cuts += 1;
      }
    }
    assert_eq!(cuts, 0, "min_duration should suppress all cuts within 1s");
  }

  #[test]
  fn clear_resets_stream_state() {
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let (a, b) = ortho_halves_frames();

    // Video 1: prime, then cut.
    assert!(det.process(make_frame(&a, 128, 96, 0)).is_none());
    let cut1 = det.process(make_frame(&b, 128, 96, 33));
    assert!(cut1.is_some());
    assert!(det.last_distance().is_some());

    det.clear();

    // First frame of video 2: no cut, state re-seeded.
    assert!(det.process(make_frame(&a, 128, 96, 1_000_000)).is_none());
    assert!(
      det.last_distance().is_none(),
      "last_distance should be cleared"
    );

    // Second frame of video 2: normal cut detection resumes.
    let cut2 = det.process(make_frame(&b, 128, 96, 1_000_033));
    assert!(cut2.is_some());
  }

  #[test]
  fn clear_preserves_resize_table_when_dims_match() {
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let (a, _) = ortho_halves_frames();
    // First frame builds the resize table for 128×96.
    det.process(make_frame(&a, 128, 96, 0));
    assert_eq!(det.resize_table.src_w, 128);
    assert_eq!(det.resize_table.src_h, 96);
    let x_offsets_len = det.resize_table.x_offsets.len();

    det.clear();
    // Table is preserved across clear — same dims on next video won't rebuild.
    assert_eq!(det.resize_table.src_w, 128);
    assert_eq!(det.resize_table.src_h, 96);
    assert_eq!(det.resize_table.x_offsets.len(), x_offsets_len);
  }

  #[test]
  fn hash_bit_packing_matches_layout() {
    // A small sanity check that bit 0 corresponds to position (0,0) and
    // higher bits walk across rows.
    let mut det = Detector::new(Options::default());
    let size = det.size;
    // Craft a known low_freq pattern: alternating above/below median.
    for i in 0..(size * size) {
      det.low_freq[i] = if i % 2 == 0 { -1.0 } else { 1.0 };
    }
    // Invoke bit-packing logic by mimicking the tail of compute_hash.
    det.sort_scratch.clone_from(&det.low_freq);
    det.sort_scratch.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = det.sort_scratch.len();
    let median = (det.sort_scratch[n / 2 - 1] + det.sort_scratch[n / 2]) / 2.0;
    det.current_hash.fill(0);
    for (i, &v) in det.low_freq.iter().enumerate() {
      if v > median {
        det.current_hash[i / 64] |= 1u64 << (i % 64);
      }
    }
    // Every odd index should be set.
    let set: u32 = det.current_hash.iter().map(|w| w.count_ones()).sum();
    assert_eq!(set as usize, size * size / 2);
  }

  #[test]
  fn options_accessors_builders_setters_roundtrip() {
    let fps30 = Timebase::new(30, nz32(1));

    let opts = Options::default()
      .with_threshold(0.5)
      .with_size(32)
      .with_lowpass(4)
      .with_min_duration(core::time::Duration::from_millis(333))
      .with_allow_initial_cut(false);
    assert_eq!(opts.threshold(), 0.5);
    assert_eq!(opts.size(), 32);
    assert_eq!(opts.lowpass(), 4);
    assert_eq!(opts.min_duration(), core::time::Duration::from_millis(333));
    assert!(!opts.allow_initial_cut());

    let opts_frames = Options::default().with_min_frames(15, fps30);
    assert_eq!(
      opts_frames.min_duration(),
      core::time::Duration::from_millis(500)
    );

    // In-place setters, chainable.
    let mut opts = Options::default();
    opts
      .set_threshold(0.1)
      .set_size(8)
      .set_lowpass(2)
      .set_min_duration(core::time::Duration::from_secs(1))
      .set_allow_initial_cut(true);
    assert_eq!(opts.threshold(), 0.1);
    assert_eq!(opts.size(), 8);
    assert_eq!(opts.lowpass(), 2);
    assert!(opts.allow_initial_cut());

    opts.set_min_frames(30, fps30);
    assert_eq!(opts.min_duration(), core::time::Duration::from_secs(1));
  }

  #[test]
  fn try_new_rejects_imsize_squared_overflow() {
    // imsize = size * lowpass = 100_000 * 100_000 = 1e10 fits in usize on
    // 64-bit. imsize^2 = 1e20 > usize::MAX (≈1.8e19) → DimensionsOverflow.
    let opts = Options::default().with_size(100_000).with_lowpass(100_000);
    let err = Detector::try_new(opts).expect_err("imsize*imsize should overflow");
    assert_eq!(
      err,
      Error::DimensionsOverflow {
        size: 100_000,
        lowpass: 100_000,
      },
    );
  }

  #[test]
  fn median_f32_singleton() {
    let mut buf = [42.0f32];
    assert_eq!(super::median_f32(&mut buf), 42.0);
  }
}
