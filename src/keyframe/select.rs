//! Keyframe selection state machine.
//!
//! Buffers per-frame [`FrameMetrics`] as they stream in, then — when the
//! caller confirms a shot boundary — partitions the buffered metrics into
//! N time-uniform buckets and emits the sharpest frame per bucket.
//!
//! # Pipeline
//!
//! 1. [`Detector::observe`] — called on every scored frame. O(1) append
//!    to an internal [`VecDeque`].
//! 2. [`Detector::finalize_shot`] — called once the application has
//!    confirmed a shot boundary (typically from a merged stream of cuts
//!    produced by the scene-detector stack). Drains every buffered entry
//!    with `ts ∈ [range.start, range.end)`, buckets them, and returns
//!    the winning timestamps in PTS order. Entries strictly older than
//!    `range.start` are discarded (they belonged to an earlier shot that
//!    was never finalised).
//! 3. [`Detector::clear`] — resets the buffer between videos.
//!
//! # Bucketing
//!
//! `N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)`
//!
//! Buckets are equal-width time slices of the shot. The first and last
//! buckets are shrunk inward by `margin_ratio · duration` to protect
//! against dissolve tails and ±2-frame scene-detector slop. If the
//! margin eats the entire bucket (large `margin_ratio` on a shot with
//! only one bucket) the detector falls back to the un-shrunk range so
//! the bucket still has a chance to emit.
//!
//! # Strict vs. fallback selection
//!
//! Per bucket, the detector tracks two running argmaxes:
//!
//! - **strict**: argmax over frames passing every hard gate (luma mean,
//!   saturation / luma variance AND-gate, clipping, minimum sharpness).
//! - **fallback**: argmax over all frames in the bucket, regardless of
//!   gate outcome.
//!
//! The emitted timestamp is the strict winner when non-empty; otherwise
//! the fallback winner. This mirrors the reference Python algorithm's
//! "prefer a degraded frame over a gap in temporal coverage" policy —
//! VLMs downstream benefit more from regular temporal sampling than from
//! the occasional missing keyframe.

use core::{cmp::Ordering, time::Duration};

use std::{collections::VecDeque, vec::Vec};

use crate::{
  frame::{TimeRange, Timestamp},
  keyframe::metrics::FrameMetrics,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---- CompositeWeights ------------------------------------------------------

/// Weights and per-metric normalisers consumed by
/// `composite_quality` when ranking frames inside a bucket.
///
/// Defaults are tuned so a "good" baseline frame (sharp, mid-luma,
/// not noisy, mildly colorful) scores ≈ 1.0. Zeroing every term
/// except `sharpness` collapses the composite to pure Tenengrad —
/// identical to the legacy strict-pass argmax.
///
/// Fields are private; use `with_*` builders for construction and
/// the field-name getters for read access.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "CompositeWeightsRaw"))]
pub struct CompositeWeights {
  sharpness: f32,
  sharpness_norm: f32,
  noise: f32,
  noise_norm: f32,
  colorfulness: f32,
  colorfulness_norm: f32,
  clipping: f32,
  motion_blur: f32,
}

impl Default for CompositeWeights {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

// Default normalisers exposed as named constants so the `with_*`
// builders can fall back to them when given an invalid (zero,
// negative, NaN, or Inf) `norm` argument. Kept in sync with the
// initialisers in [`CompositeWeights::new`].
const DEFAULT_SHARPNESS_NORM: f32 = 1000.0;
const DEFAULT_NOISE_NORM: f32 = 20.0;
const DEFAULT_COLORFULNESS_NORM: f32 = 50.0;

/// Returns `norm` when it is strictly positive and finite, otherwise
/// returns `default`. Invalid normalisers would feed `Inf`/`NaN` into
/// [`composite_quality`] and silently corrupt the strict-pass argmax
/// (the NaN-tolerant [`sharper`] helper retains the first non-numeric
/// incumbent).
#[inline]
const fn sanitise_norm(norm: f32, default: f32) -> f32 {
  if norm.is_finite() && norm > 0.0 {
    norm
  } else {
    default
  }
}

/// Returns `weight` when it is finite (positive, zero, or negative all
/// allowed — only non-finite values are filtered). Non-finite weights
/// collapse to `0.0` so the term contributes nothing rather than
/// propagating `Inf`/`NaN` through [`composite_quality`]. Negative
/// weights are preserved: a user can deliberately invert a term's
/// sense (e.g. rewarding clipping for an unusual workload).
#[inline]
const fn sanitise_weight(weight: f32) -> f32 {
  if weight.is_finite() { weight } else { 0.0 }
}

// Private deserialization shim for [`CompositeWeights`]. Routes every
// weight through [`sanitise_weight`] and every normaliser through
// [`sanitise_norm`] so a serialized configuration with invalid
// (`NaN`/`Inf` weights, or zero/negative/`NaN`/`Inf` norms) cannot
// reach [`composite_quality`] and silently corrupt the strict-pass
// argmax. The struct mirrors [`CompositeWeights`]'s field shape
// exactly — adding a field on [`CompositeWeights`] requires updating
// both this struct and the [`From`] impl below.
#[cfg(feature = "serde")]
#[derive(Deserialize)]
struct CompositeWeightsRaw {
  sharpness: f32,
  sharpness_norm: f32,
  noise: f32,
  noise_norm: f32,
  colorfulness: f32,
  colorfulness_norm: f32,
  clipping: f32,
  motion_blur: f32,
}

#[cfg(feature = "serde")]
impl From<CompositeWeightsRaw> for CompositeWeights {
  #[inline]
  fn from(r: CompositeWeightsRaw) -> Self {
    Self {
      sharpness: sanitise_weight(r.sharpness),
      sharpness_norm: sanitise_norm(r.sharpness_norm, DEFAULT_SHARPNESS_NORM),
      noise: sanitise_weight(r.noise),
      noise_norm: sanitise_norm(r.noise_norm, DEFAULT_NOISE_NORM),
      colorfulness: sanitise_weight(r.colorfulness),
      colorfulness_norm: sanitise_norm(r.colorfulness_norm, DEFAULT_COLORFULNESS_NORM),
      clipping: sanitise_weight(r.clipping),
      motion_blur: sanitise_weight(r.motion_blur),
    }
  }
}

impl CompositeWeights {
  /// Creates a [`CompositeWeights`] with the calibrated default
  /// weights and normalisers. See the type docs for the calibration
  /// rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sharpness: 1.0,
      sharpness_norm: DEFAULT_SHARPNESS_NORM,
      noise: 0.3,
      noise_norm: DEFAULT_NOISE_NORM,
      colorfulness: 0.2,
      colorfulness_norm: DEFAULT_COLORFULNESS_NORM,
      clipping: 0.5,
      motion_blur: 0.0,
    }
  }

  /// Sets the sharpness weight and its normaliser.
  ///
  /// Invalid normalisers (zero, negative, `NaN`, or infinite) are
  /// silently clamped to the [`new`](Self::new) default
  /// (`1000.0`). They would otherwise feed `Inf`/`NaN` into
  /// [`composite_quality`] and silently corrupt the strict-pass
  /// argmax (the NaN-tolerant [`sharper`] helper retains the first
  /// non-numeric incumbent and later candidates cannot unseat it).
  /// The weight itself is stored verbatim — passing `weight = 0.0`
  /// is the right way to disable a term.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sharpness(mut self, weight: f32, norm: f32) -> Self {
    self.sharpness = sanitise_weight(weight);
    self.sharpness_norm = sanitise_norm(norm, DEFAULT_SHARPNESS_NORM);
    self
  }
  /// Sets the noise weight and its normaliser. Noise is a penalty
  /// (subtracted in the composite).
  ///
  /// Invalid normalisers are silently clamped to the
  /// [`new`](Self::new) default (`20.0`); see
  /// [`Self::with_sharpness`] for the rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_noise(mut self, weight: f32, norm: f32) -> Self {
    self.noise = sanitise_weight(weight);
    self.noise_norm = sanitise_norm(norm, DEFAULT_NOISE_NORM);
    self
  }
  /// Sets the colorfulness weight and its normaliser.
  ///
  /// Invalid normalisers are silently clamped to the
  /// [`new`](Self::new) default (`50.0`); see
  /// [`Self::with_sharpness`] for the rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_colorfulness(mut self, weight: f32, norm: f32) -> Self {
    self.colorfulness = sanitise_weight(weight);
    self.colorfulness_norm = sanitise_norm(norm, DEFAULT_COLORFULNESS_NORM);
    self
  }
  /// Sets the clipping-penalty weight. Clipping is already in `[0, 1]`
  /// — no normaliser.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_clipping(mut self, weight: f32) -> Self {
    self.clipping = sanitise_weight(weight);
    self
  }
  /// Sets the motion-blur-penalty weight. Anisotropy is already in
  /// `[0, 1]` — no normaliser. Defaults to 0 (off).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur(mut self, weight: f32) -> Self {
    self.motion_blur = sanitise_weight(weight);
    self
  }

  // ---- Getters ------------------------------------------------------------

  /// Sharpness weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness(&self) -> f32 {
    self.sharpness
  }
  /// Sharpness normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness_norm(&self) -> f32 {
    self.sharpness_norm
  }
  /// Noise weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise(&self) -> f32 {
    self.noise
  }
  /// Noise normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise_norm(&self) -> f32 {
    self.noise_norm
  }
  /// Colorfulness weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness(&self) -> f32 {
    self.colorfulness
  }
  /// Colorfulness normaliser (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness_norm(&self) -> f32 {
    self.colorfulness_norm
  }
  /// Clipping-penalty weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clipping(&self) -> f32 {
    self.clipping
  }
  /// Motion-blur-penalty weight (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur(&self) -> f32 {
    self.motion_blur
  }
}

// ---- Options ---------------------------------------------------------------

/// Tuning knobs for the selection state machine.
///
/// Default values are calibrated for 256-px longest-side preprocessed
/// input (the default of [`crate::keyframe::preprocess::Downscaler`]).
/// Changing the downscale dimension invalidates the sharpness and
/// variance thresholds — keep them in sync.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  target_interval: Duration,
  max_frames_per_shot: u32,
  margin_ratio: f64,
  min_sharpness: f32,
  black_mean_threshold: u8,
  bright_mean_threshold: u8,
  luma_variance_threshold: f32,
  sat_variance_threshold: f32,
  max_clipping: f32,
  weights: CompositeWeights,
  adaptive_floor: bool,
  adaptive_floor_percentile: f32,
  adaptive_floor_min_samples: usize,
  motion_blur_gate: bool,
  max_motion_blur: f32,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self {
      target_interval: Duration::from_secs(4),
      max_frames_per_shot: 16,
      margin_ratio: 0.02,
      min_sharpness: 100.0,
      black_mean_threshold: 15,
      bright_mean_threshold: 240,
      luma_variance_threshold: 5.0,
      sat_variance_threshold: 3.0,
      max_clipping: 0.5,
      weights: CompositeWeights::new(),
      adaptive_floor: true,
      adaptive_floor_percentile: 0.25,
      adaptive_floor_min_samples: 20,
      motion_blur_gate: false,
      max_motion_blur: 0.75,
    }
  }
}

impl Options {
  /// Creates a new [`Options`] matching [`Options::default`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new() -> Self {
    Self::default()
  }

  /// Target interval between keyframes. Drives the bucket count via
  /// `ceil(shot_duration / target_interval)`.
  ///
  /// # Panics
  /// When `d` is [`Duration::ZERO`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_target_interval(mut self, d: Duration) -> Self {
    assert!(!d.is_zero(), "target_interval must be > 0");
    self.target_interval = d;
    self
  }

  /// Upper bound on the number of keyframes emitted per shot.
  ///
  /// # Panics
  /// When `n == 0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_frames_per_shot(mut self, n: u32) -> Self {
    assert!(n > 0, "max_frames_per_shot must be > 0");
    self.max_frames_per_shot = n;
    self
  }

  /// Fraction of shot duration to shrink away from the first and last
  /// buckets.
  ///
  /// # Panics
  /// When outside `[0.0, 0.5)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_margin_ratio(mut self, r: f64) -> Self {
    assert!(
      (0.0..0.5).contains(&r),
      "margin_ratio must be in [0.0, 0.5)"
    );
    self.margin_ratio = r;
    self
  }

  /// Minimum Tenengrad sharpness for the strict pass.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_sharpness(mut self, s: f32) -> Self {
    self.min_sharpness = s;
    self
  }

  /// Luma-mean floor — frames below this are flagged too-dark.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_black_mean_threshold(mut self, t: u8) -> Self {
    self.black_mean_threshold = t;
    self
  }

  /// Luma-mean ceiling — frames above this are flagged overexposed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_bright_mean_threshold(mut self, t: u8) -> Self {
    self.bright_mean_threshold = t;
    self
  }

  /// Luma-variance floor. AND-gated with
  /// [`Self::with_sat_variance_threshold`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_luma_variance_threshold(mut self, t: f32) -> Self {
    self.luma_variance_threshold = t;
    self
  }

  /// Saturation-variance floor. AND-gated with
  /// [`Self::with_luma_variance_threshold`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sat_variance_threshold(mut self, t: f32) -> Self {
    self.sat_variance_threshold = t;
    self
  }

  /// Maximum tolerated clipping ratio.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_max_clipping(mut self, c: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&c),
      "max_clipping must be in [0.0, 1.0]"
    );
    self.max_clipping = c;
    self
  }

  /// Replaces the [`CompositeWeights`] driving the strict-pass argmax
  /// inside [`Detector::finalize_shot`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_composite_weights(mut self, w: CompositeWeights) -> Self {
    self.weights = w;
    self
  }

  /// Enables or disables the adaptive per-shot sharpness floor. When
  /// enabled (the default), the effective strict-gate sharpness floor
  /// becomes `min(min_sharpness, p_in_shot)` for shots that meet
  /// [`Self::adaptive_floor_min_samples`] — the absolute floor is
  /// only **lowered**, never raised.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_adaptive_floor(mut self, on: bool) -> Self {
    self.adaptive_floor = on;
    self
  }

  /// Sets the percentile (in `[0.0, 1.0]`) used by the adaptive floor.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_adaptive_floor_percentile(mut self, p: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&p),
      "adaptive_floor_percentile must be in [0.0, 1.0]"
    );
    self.adaptive_floor_percentile = p;
    self
  }

  /// Sets the minimum in-shot sample count required to activate the
  /// adaptive floor. Below this, the absolute floor is used unchanged.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_adaptive_floor_min_samples(mut self, n: usize) -> Self {
    self.adaptive_floor_min_samples = n;
    self
  }

  /// Enables or disables the motion-blur hard gate. When enabled,
  /// frames whose [`FrameMetrics::motion_blur`] exceeds
  /// [`Self::max_motion_blur`] are rejected from the strict pass.
  /// Off by default because gradient anisotropy at 256-px downscale
  /// confounds motion blur with single-orientation scenes (forest,
  /// façade, horizon).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur_gate(mut self, on: bool) -> Self {
    self.motion_blur_gate = on;
    self
  }

  /// Sets the maximum tolerated motion-blur (anisotropy) score. The
  /// gate (when enabled) rejects frames strictly greater than this
  /// value.
  ///
  /// # Panics
  /// When outside `[0.0, 1.0]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_max_motion_blur(mut self, m: f32) -> Self {
    assert!(
      (0.0..=1.0).contains(&m),
      "max_motion_blur must be in [0.0, 1.0]"
    );
    self.max_motion_blur = m;
    self
  }

  /// Whether the motion-blur hard gate is enabled (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur_gate(&self) -> bool {
    self.motion_blur_gate
  }
  /// Maximum tolerated motion-blur anisotropy score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_motion_blur(&self) -> f32 {
    self.max_motion_blur
  }

  /// Whether the adaptive per-shot sharpness floor is enabled (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor(&self) -> bool {
    self.adaptive_floor
  }
  /// Adaptive floor percentile (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor_percentile(&self) -> f32 {
    self.adaptive_floor_percentile
  }
  /// Adaptive floor minimum-samples threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_floor_min_samples(&self) -> usize {
    self.adaptive_floor_min_samples
  }

  /// Composite-quality weights and normalisers (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn composite_weights(&self) -> &CompositeWeights {
    &self.weights
  }

  /// Target inter-keyframe interval (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn target_interval(&self) -> Duration {
    self.target_interval
  }
  /// Maximum keyframes per shot (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_frames_per_shot(&self) -> u32 {
    self.max_frames_per_shot
  }
  /// First/last-bucket margin fraction (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn margin_ratio(&self) -> f64 {
    self.margin_ratio
  }
  /// Strict-pass sharpness floor (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_sharpness(&self) -> f32 {
    self.min_sharpness
  }
  /// Black-frame luma-mean threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn black_mean_threshold(&self) -> u8 {
    self.black_mean_threshold
  }
  /// Overexposed luma-mean threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bright_mean_threshold(&self) -> u8 {
    self.bright_mean_threshold
  }
  /// Luma-variance threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_variance_threshold(&self) -> f32 {
    self.luma_variance_threshold
  }
  /// Saturation-variance threshold (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sat_variance_threshold(&self) -> f32 {
    self.sat_variance_threshold
  }
  /// Max clipping ratio (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_clipping(&self) -> f32 {
    self.max_clipping
  }
}

// ---- Detector --------------------------------------------------------------

/// The keyframe selection state machine.
#[derive(Debug, Clone)]
pub struct Detector {
  opts: Options,
  buffer: VecDeque<(Timestamp, FrameMetrics)>,
}

impl Detector {
  /// Creates a new detector with the supplied options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: Options) -> Self {
    Self {
      opts,
      buffer: VecDeque::new(),
    }
  }

  /// Returns the detector's current options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.opts
  }

  /// Number of scored frames currently buffered (awaiting finalization).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn buffered(&self) -> usize {
    self.buffer.len()
  }

  /// Appends a scored frame to the buffer. Callers must feed frames in
  /// non-decreasing PTS order.
  ///
  /// In debug builds, an out-of-order call panics via `debug_assert!`.
  /// In release builds the precondition is not checked; an out-of-order
  /// entry reaches the bucket-walker in [`Self::finalize_shot`] and is
  /// silently dropped (either because it predates the cursor's current
  /// bucket start, or because it lands in a gap between buckets). This
  /// is the intended failure mode — keyframe selection prefers to
  /// tolerate minor ordering slop over aborting mid-stream — but if
  /// you care about catching decoder bugs, run with debug assertions
  /// enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn observe(&mut self, ts: Timestamp, metrics: FrameMetrics) {
    debug_assert!(
      self
        .buffer
        .back()
        .is_none_or(|(prev, _)| prev.cmp_semantic(&ts) != Ordering::Greater),
      "observe() frames must arrive in non-decreasing PTS order"
    );
    self.buffer.push_back((ts, metrics));
  }

  /// A shot boundary has been confirmed. Drains every buffered entry
  /// whose timestamp lies in `[range.start(), range.end())`, buckets
  /// them, and returns the winning timestamp per bucket in PTS order.
  ///
  /// Entries strictly older than `range.start()` are silently discarded
  /// (they belonged to an earlier shot that was never finalized, or
  /// were observed before the first shot began).
  ///
  /// A degenerate or reversed `range` (duration ≤ 0) yields an empty
  /// result and still drops stale entries.
  ///
  /// Returns an owned `Vec<Timestamp>` rather than a borrowing iterator.
  /// Size is bounded by
  /// [`Options::max_frames_per_shot`](Options::with_max_frames_per_shot) —
  /// typically ≤ 16 entries, so the allocation is small. Caller may hold
  /// the result across subsequent detector calls.
  pub fn finalize_shot(&mut self, range: TimeRange) -> Vec<Timestamp> {
    // 1. Drop stale entries (before the shot starts).
    while let Some((ts, _)) = self.buffer.front() {
      if ts.cmp_semantic(&range.start()) == Ordering::Less {
        self.buffer.pop_front();
      } else {
        break;
      }
    }

    // 2. Degenerate range (zero or negative duration) → nothing to emit.
    let duration = match range.duration() {
      Some(d) if !d.is_zero() => d,
      _ => return Vec::new(),
    };

    // 2.5. Compute the effective strict-gate sharpness floor for this
    // shot. If adaptive_floor is enabled and the shot has at least
    // `adaptive_floor_min_samples` buffered in-range entries, set the
    // floor to `min(absolute_floor, p_percentile)` — never raising the
    // floor. This lets legitimate low-detail shots (fog, night
    // interiors) produce strict winners instead of always degrading
    // to fallback selection.
    let effective_min_sharpness = compute_effective_floor(&self.buffer, &range, &self.opts);

    // 3. Compute bucket count and precompute per-bucket effective
    //    [start, end) timestamps (with first-/last-bucket margin shrink).
    let n = compute_n_buckets(duration, &self.opts);
    let bucket_ranges = compute_bucket_ranges(&range, n, self.opts.margin_ratio);

    // 4. Single linear walk across the in-range entries. Entries inside
    //    the first-bucket's pre-margin or last-bucket's post-margin zones
    //    are skipped. Track strict + fallback running argmaxes per bucket.
    let mut emits: Vec<Timestamp> = Vec::with_capacity(n);
    let mut bucket_idx = 0usize;
    let mut best_strict: Option<(Timestamp, f32)> = None;
    let mut best_any: Option<(Timestamp, f32)> = None;

    // Snapshot opts locally to avoid borrowing self while draining the
    // buffer.
    let opts = self.opts;

    while let Some((ts, metrics)) = self.buffer.front().copied() {
      if ts.cmp_semantic(&range.end()) != Ordering::Less {
        break; // past this shot — leave for the next finalize.
      }

      // Advance bucket cursor while the entry is past the current
      // bucket's effective end. Each advance emits the previous
      // bucket's winner.
      while bucket_idx < n && ts.cmp_semantic(&bucket_ranges[bucket_idx].1) != Ordering::Less {
        if let Some((t, _)) = best_strict.or(best_any) {
          emits.push(t);
        }
        best_strict = None;
        best_any = None;
        bucket_idx += 1;
      }

      // Consume the entry from the buffer. We're inside the shot.
      self.buffer.pop_front();

      if bucket_idx >= n {
        // Past the last bucket's effective end, but still inside the
        // shot (i.e. within the last-bucket margin zone). Drain but do
        // not score.
        continue;
      }

      // Skip entries that fall in the first bucket's pre-margin gap.
      if ts.cmp_semantic(&bucket_ranges[bucket_idx].0) == Ordering::Less {
        continue;
      }

      // Running argmax updates.
      // Fallback path: pure-sharpness ranking, preserved so "least
      // bad" is well-defined when every frame in the bucket fails
      // the strict gate.
      if best_any.is_none_or(|(_, s)| sharper(metrics.sharpness(), s)) {
        best_any = Some((ts, metrics.sharpness()));
      }
      // Strict path: composite-quality ranking among gate-passing
      // frames. Non-finite composites are skipped — a NaN incumbent
      // would lock out every later finite candidate because
      // `sharper(finite, NaN) == false`. Weights and norms are already
      // sanitised at the [`CompositeWeights`] boundary, but
      // [`FrameMetrics`] setters accept arbitrary `f32` values and
      // detector kernels could conceivably produce `Inf` on
      // pathological inputs (e.g. an integer accumulator that
      // saturated). Filtering here is the one safety net the argmax
      // needs.
      if !hard_gate(&metrics, &opts) && metrics.sharpness() >= effective_min_sharpness {
        let q = composite_quality(&metrics, opts.composite_weights());
        if q.is_finite() && best_strict.is_none_or(|(_, s)| sharper(q, s)) {
          best_strict = Some((ts, q));
        }
      }
    }

    // Flush the last active bucket.
    if bucket_idx < n {
      if let Some((t, _)) = best_strict.or(best_any) {
        emits.push(t);
      }
    }

    emits
  }

  /// Closes out any buffered entries as a "final shot" ending at `eos`.
  ///
  /// Convenience wrapper for end-of-stream handling: forms a synthetic
  /// [`TimeRange`] that starts at the oldest buffered entry's timestamp
  /// and ends at `eos`, then calls [`Self::finalize_shot`]. Returns an
  /// empty `Vec` when the buffer is empty or when `eos` is not strictly
  /// after the earliest buffered entry.
  ///
  /// Callers that track the previous-cut timestamp themselves can
  /// equivalently call [`Self::finalize_shot`] directly with a range
  /// they construct.
  pub fn finalize_remaining(&mut self, eos: Timestamp) -> Vec<Timestamp> {
    let Some(&(first_ts, _)) = self.buffer.front() else {
      return Vec::new();
    };
    // Re-express the start timestamp in eos's timebase so the TimeRange
    // constructor (which takes raw pts + a single timebase) sees a
    // self-consistent pair.
    let tb = eos.timebase();
    let start_pts = first_ts.rescale_to(tb).pts();
    let end_pts = eos.pts();
    if end_pts <= start_pts {
      return Vec::new();
    }
    self.finalize_shot(TimeRange::new(start_pts, end_pts, tb))
  }

  /// Resets the detector's buffer. The configured options are preserved.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.buffer.clear();
  }
}

// ---- Helpers ---------------------------------------------------------------

/// `N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)`
///
/// Uses integer (nanosecond) arithmetic to avoid pulling in `f64::ceil`,
/// which isn't available in `no_std` builds.
fn compute_n_buckets(duration: Duration, opts: &Options) -> usize {
  let target_ns = opts.target_interval.as_nanos();
  if target_ns == 0 {
    return 1;
  }
  // Ceiling division in u128.
  let d_ns = duration.as_nanos();
  let raw = (d_ns.saturating_add(target_ns - 1)) / target_ns;
  let capped = raw.min(opts.max_frames_per_shot as u128).max(1);
  capped as usize
}

/// Returns the `(effective_start, effective_end)` timestamp of each
/// bucket, margin applied to the first and last. When the margin eats
/// the whole bucket (degenerate configuration), falls back to the
/// un-shrunk bucket so the bucket still has a chance to contribute.
fn compute_bucket_ranges(
  range: &TimeRange,
  n: usize,
  margin_ratio: f64,
) -> Vec<(Timestamp, Timestamp)> {
  let mut out = Vec::with_capacity(n);
  let n_f = n as f64;
  for b in 0..n {
    let t0 = (b as f64) / n_f;
    let t1 = ((b + 1) as f64) / n_f;
    let eff_t0 = if b == 0 { t0 + margin_ratio } else { t0 };
    let eff_t1 = if b == n - 1 { t1 - margin_ratio } else { t1 };
    let (use_t0, use_t1) = if eff_t1 > eff_t0 {
      (eff_t0, eff_t1)
    } else {
      (t0, t1) // un-shrunk fallback
    };
    out.push((range.interpolate(use_t0), range.interpolate(use_t1)));
  }
  out
}

/// Returns the effective strict-gate sharpness floor for the shot
/// described by `range`, given the entries currently buffered and the
/// adaptive-floor options. Never raises the floor above
/// [`Options::min_sharpness`]; only lowers it.
fn compute_effective_floor(
  buffer: &VecDeque<(Timestamp, FrameMetrics)>,
  range: &TimeRange,
  opts: &Options,
) -> f32 {
  if !opts.adaptive_floor() {
    return opts.min_sharpness();
  }
  // Collect in-range sharpness values.
  let mut sharps: Vec<f32> = buffer
    .iter()
    .filter(|(ts, _)| {
      ts.cmp_semantic(&range.start()) != Ordering::Less
        && ts.cmp_semantic(&range.end()) == Ordering::Less
    })
    .map(|(_, m)| m.sharpness())
    .collect();
  // `min_samples == 0` (a valid user setting meaning "adapt regardless
  // of sample count") combined with an empty in-range shot would skip
  // the threshold check and then index an empty slice. Treat
  // `sharps.is_empty()` as "no data to derive a percentile from" and
  // fall back to the absolute floor.
  if sharps.is_empty() || sharps.len() < opts.adaptive_floor_min_samples() {
    return opts.min_sharpness();
  }
  sharps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
  let idx = ((sharps.len() as f32 * opts.adaptive_floor_percentile()) as usize)
    .min(sharps.len().saturating_sub(1));
  let p = sharps[idx];
  opts.min_sharpness().min(p) // never raise the floor
}

/// Weighted composite of [`FrameMetrics`] used as the strict-pass
/// argmax key inside [`Detector::finalize_shot`].
///
/// Higher is better. `sharpness` and `colorfulness` contribute
/// positively; `noise`, `clipping`, and `motion_blur` contribute
/// negatively. The fallback path inside `finalize_shot` still ranks
/// by raw `sharpness` — see the module docs.
#[cfg_attr(not(tarpaulin), inline(always))]
fn composite_quality(m: &FrameMetrics, w: &CompositeWeights) -> f32 {
  let s = w.sharpness() * (m.sharpness() / w.sharpness_norm());
  let n = w.noise() * (m.noise() / w.noise_norm());
  let c = w.colorfulness() * (m.colorfulness() / w.colorfulness_norm());
  let clip = w.clipping() * m.clipping();
  let mb = w.motion_blur() * m.motion_blur();
  s - n + c - clip - mb
}

/// `true` when `a > b` under `f32::partial_cmp`. NaN compares as not-
/// greater, so a NaN score cannot unseat a numeric incumbent — our
/// reductions do not produce NaN anyway, but the defensive default
/// keeps the running-argmax state well-defined under any input.
#[cfg_attr(not(tarpaulin), inline(always))]
fn sharper(a: f32, b: f32) -> bool {
  a.partial_cmp(&b).unwrap_or(Ordering::Equal) == Ordering::Greater
}

/// Any-one-trips hard gate matching the Python reference's flat /
/// over-/under-exposed / clipped checks.
fn hard_gate(m: &FrameMetrics, opts: &Options) -> bool {
  if m.brightness() < opts.black_mean_threshold as f32 {
    return true;
  }
  if m.brightness() > opts.bright_mean_threshold as f32 {
    return true;
  }
  // AND-gate: only flag flat when BOTH variances are low (keeps
  // equiluminant multi-colour frames).
  if m.luma_variance() < opts.luma_variance_threshold
    && m.saturation_variance() < opts.sat_variance_threshold
  {
    return true;
  }
  if m.clipping() > opts.max_clipping {
    return true;
  }
  if opts.motion_blur_gate && m.motion_blur() > opts.max_motion_blur {
    return true;
  }
  false
}

// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::frame::{Timebase, Timestamp};
  use core::num::NonZeroU32;

  fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).expect("non-zero")
  }

  fn ts(pts: i64) -> Timestamp {
    // 1 µs per tick — makes pts directly readable as microseconds.
    Timestamp::new(pts, Timebase::new(1, nz(1_000_000)))
  }

  fn tr(start_us: i64, end_us: i64) -> TimeRange {
    TimeRange::new(start_us, end_us, Timebase::new(1, nz(1_000_000)))
  }

  fn good_metrics(sharpness: f32) -> FrameMetrics {
    FrameMetrics::new()
      .with_sharpness(sharpness)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
  }

  // ----- Options -------------------------------------------------------------

  #[test]
  fn default_options_match_design_doc() {
    let o = Options::default();
    assert_eq!(o.target_interval(), Duration::from_secs(4));
    assert_eq!(o.max_frames_per_shot(), 16);
    assert!((o.margin_ratio() - 0.02).abs() < 1e-9);
    assert_eq!(o.min_sharpness(), 100.0);
    assert_eq!(o.black_mean_threshold(), 15);
    assert_eq!(o.bright_mean_threshold(), 240);
    assert_eq!(o.luma_variance_threshold(), 5.0);
    assert_eq!(o.sat_variance_threshold(), 3.0);
    assert_eq!(o.max_clipping(), 0.5);
  }

  #[test]
  fn options_builders_roundtrip() {
    let o = Options::new()
      .with_target_interval(Duration::from_secs(2))
      .with_max_frames_per_shot(8)
      .with_margin_ratio(0.05)
      .with_min_sharpness(50.0)
      .with_black_mean_threshold(10)
      .with_bright_mean_threshold(245)
      .with_luma_variance_threshold(1.0)
      .with_sat_variance_threshold(2.0)
      .with_max_clipping(0.25);
    assert_eq!(o.target_interval(), Duration::from_secs(2));
    assert_eq!(o.max_frames_per_shot(), 8);
    assert_eq!(o.margin_ratio(), 0.05);
    assert_eq!(o.min_sharpness(), 50.0);
    assert_eq!(o.black_mean_threshold(), 10);
    assert_eq!(o.bright_mean_threshold(), 245);
    assert_eq!(o.luma_variance_threshold(), 1.0);
    assert_eq!(o.sat_variance_threshold(), 2.0);
    assert_eq!(o.max_clipping(), 0.25);
  }

  #[test]
  #[should_panic(expected = "max_frames_per_shot")]
  fn options_max_frames_zero_panics() {
    let _ = Options::new().with_max_frames_per_shot(0);
  }

  #[test]
  #[should_panic(expected = "margin_ratio")]
  fn options_margin_half_panics() {
    let _ = Options::new().with_margin_ratio(0.5);
  }

  #[test]
  #[should_panic(expected = "max_clipping")]
  fn options_max_clipping_out_of_range_panics() {
    let _ = Options::new().with_max_clipping(1.5);
  }

  // ----- compute_n_buckets ---------------------------------------------------

  #[test]
  fn n_buckets_scales_with_duration() {
    let o = Options::default();
    assert_eq!(compute_n_buckets(Duration::from_secs(1), &o), 1);
    assert_eq!(compute_n_buckets(Duration::from_secs(4), &o), 1);
    assert_eq!(compute_n_buckets(Duration::from_secs(5), &o), 2);
    assert_eq!(compute_n_buckets(Duration::from_secs(12), &o), 3);
    assert_eq!(compute_n_buckets(Duration::from_secs(16), &o), 4);
  }

  #[test]
  fn n_buckets_capped_by_max_frames() {
    let o = Options::default().with_max_frames_per_shot(5);
    // 100s with target 4s would ask for 25; clamped to 5.
    assert_eq!(compute_n_buckets(Duration::from_secs(100), &o), 5);
  }

  #[test]
  fn n_buckets_floor_is_one() {
    let o = Options::default();
    assert_eq!(compute_n_buckets(Duration::from_nanos(1), &o), 1);
  }

  // ----- hard_gate -----------------------------------------------------------

  #[test]
  fn hard_gate_rejects_too_dark() {
    let o = Options::default();
    let mut m = good_metrics(200.0);
    m.set_brightness(5.0);
    assert!(hard_gate(&m, &o));
  }

  #[test]
  fn hard_gate_rejects_too_bright() {
    let o = Options::default();
    let mut m = good_metrics(200.0);
    m.set_brightness(250.0);
    assert!(hard_gate(&m, &o));
  }

  #[test]
  fn hard_gate_rejects_flat_frame() {
    let o = Options::default();
    let mut m = good_metrics(200.0);
    m.set_luma_variance(1.0);
    m.set_saturation_variance(1.0);
    assert!(hard_gate(&m, &o));
  }

  #[test]
  fn hard_gate_keeps_equiluminant_multicolour() {
    // Low luma variance but high saturation variance — the AND-gate
    // keeps this frame alive.
    let o = Options::default();
    let mut m = good_metrics(200.0);
    m.set_luma_variance(1.0);
    m.set_saturation_variance(80.0);
    assert!(!hard_gate(&m, &o));
  }

  #[test]
  fn hard_gate_rejects_heavy_clipping() {
    let o = Options::default();
    let mut m = good_metrics(200.0);
    m.set_clipping(0.9);
    assert!(hard_gate(&m, &o));
  }

  // ----- Detector ------------------------------------------------------------

  #[test]
  fn observe_and_buffered() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(0), good_metrics(100.0));
    det.observe(ts(1_000), good_metrics(200.0));
    assert_eq!(det.buffered(), 2);
  }

  #[test]
  fn clear_empties_buffer() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(0), good_metrics(100.0));
    det.clear();
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_single_bucket_picks_sharpest() {
    // 2-second shot with target_interval=4s → 1 bucket.
    let opts = Options::default().with_margin_ratio(0.0); // disable margin
    let mut det = Detector::new(opts);
    det.observe(ts(0), good_metrics(100.0));
    det.observe(ts(500_000), good_metrics(500.0)); // sharpest
    det.observe(ts(1_500_000), good_metrics(200.0));

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 0, "in-range entries should be drained");
  }

  #[test]
  fn finalize_multiple_buckets_pick_per_bucket_sharpest() {
    // 12s shot with target 4s → 3 buckets; disable margin for clean bounds.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);

    // Bucket 0: [0, 4s). Best at 1s.
    det.observe(ts(500_000), good_metrics(100.0));
    det.observe(ts(1_000_000), good_metrics(300.0));
    det.observe(ts(3_500_000), good_metrics(150.0));
    // Bucket 1: [4s, 8s). Best at 5s.
    det.observe(ts(4_500_000), good_metrics(200.0));
    det.observe(ts(5_000_000), good_metrics(500.0));
    det.observe(ts(7_500_000), good_metrics(100.0));
    // Bucket 2: [8s, 12s). Best at 10s.
    det.observe(ts(9_000_000), good_metrics(150.0));
    det.observe(ts(10_000_000), good_metrics(450.0));
    det.observe(ts(11_500_000), good_metrics(200.0));

    let out = det.finalize_shot(tr(0, 12_000_000));
    assert_eq!(out, vec![ts(1_000_000), ts(5_000_000), ts(10_000_000)]);
  }

  #[test]
  fn finalize_falls_back_when_all_frames_fail_gates() {
    // Entire bucket's frames are "bad" (too dark). Fallback picks the
    // sharpest anyway.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let bad = |sharp| {
      FrameMetrics::new()
        .with_sharpness(sharp)
        .with_brightness(5.0)
        .with_luma_variance(200.0)
        .with_saturation_variance(100.0)
        .with_clipping(0.0)
    };
    det.observe(ts(0), bad(100.0));
    det.observe(ts(1_000_000), bad(400.0)); // sharpest among bad
    det.observe(ts(3_000_000), bad(200.0));

    let out = det.finalize_shot(tr(0, 4_000_000));
    assert_eq!(out, vec![ts(1_000_000)]);
  }

  #[test]
  fn finalize_skips_bucket_with_no_entries() {
    // 3-bucket shot; middle bucket has no observations.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(1_000_000), good_metrics(300.0));
    // 4..8 s: nothing
    det.observe(ts(9_000_000), good_metrics(400.0));

    let out = det.finalize_shot(tr(0, 12_000_000));
    assert_eq!(out, vec![ts(1_000_000), ts(9_000_000)]);
  }

  #[test]
  fn finalize_drops_stale_entries_from_earlier_shots() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(100), good_metrics(500.0)); // pre-shot, should be dropped
    det.observe(ts(500_000), good_metrics(200.0));

    let out = det.finalize_shot(tr(200_000, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_retains_post_shot_entries_for_next_call() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_metrics(100.0));
    det.observe(ts(5_000_000), good_metrics(900.0)); // belongs to next shot

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
    assert_eq!(det.buffered(), 1, "future entries preserved");
    let out2 = det.finalize_shot(tr(2_000_000, 6_000_000));
    assert_eq!(out2, vec![ts(5_000_000)]);
  }

  #[test]
  fn finalize_degenerate_range_returns_empty_and_drops_stale() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(100), good_metrics(100.0));
    det.observe(ts(500_000), good_metrics(200.0));

    // end == start → zero duration, no emits. Stale (pts < 200_000)
    // entries still dropped.
    let out = det.finalize_shot(tr(200_000, 200_000));
    assert!(out.is_empty());
    assert_eq!(det.buffered(), 1, "only the future entry remains");
  }

  #[test]
  fn finalize_respects_first_bucket_margin() {
    // 10-s shot, 1 bucket. Margin 0.1 → effective bucket is [1s, 9s).
    // An entry at 500 ms should be in the pre-margin gap (skipped),
    // one at 5 s used.
    let opts = Options::default()
      .with_target_interval(Duration::from_secs(20)) // force 1 bucket
      .with_margin_ratio(0.1);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_metrics(900.0)); // pre-margin
    det.observe(ts(5_000_000), good_metrics(300.0)); // in-bucket
    det.observe(ts(9_500_000), good_metrics(800.0)); // post-margin

    let out = det.finalize_shot(tr(0, 10_000_000));
    assert_eq!(out, vec![ts(5_000_000)]);
  }

  #[test]
  fn finalize_emits_in_pts_order() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(1_000_000), good_metrics(100.0));
    det.observe(ts(5_000_000), good_metrics(100.0));
    det.observe(ts(9_000_000), good_metrics(100.0));
    let out = det.finalize_shot(tr(0, 12_000_000));
    assert!(out.windows(2).all(|w| w[0].pts() < w[1].pts()));
  }

  #[test]
  fn finalize_can_be_called_multiple_times() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    det.observe(ts(500_000), good_metrics(100.0));
    det.observe(ts(5_000_000), good_metrics(100.0));
    let out1 = det.finalize_shot(tr(0, 2_000_000));
    let out2 = det.finalize_shot(tr(2_000_000, 6_000_000));
    assert_eq!(out1.len(), 1);
    assert_eq!(out2.len(), 1);
  }

  #[test]
  fn finalize_remaining_treats_buffer_tail_as_final_shot() {
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    // First shot closed normally.
    det.observe(ts(500_000), good_metrics(100.0));
    let _ = det.finalize_shot(tr(0, 2_000_000));
    // Second shot opens but EOS arrives before a confirmed cut.
    det.observe(ts(3_000_000), good_metrics(200.0));
    det.observe(ts(4_500_000), good_metrics(400.0));

    let out = det.finalize_remaining(ts(6_000_000));
    assert_eq!(out, vec![ts(4_500_000)]);
    assert_eq!(det.buffered(), 0);
  }

  #[test]
  fn finalize_remaining_empty_buffer_returns_empty() {
    let mut det = Detector::new(Options::default());
    let out = det.finalize_remaining(ts(1_000_000));
    assert!(out.is_empty());
  }

  #[test]
  fn finalize_remaining_eos_before_buffer_start_returns_empty() {
    let mut det = Detector::new(Options::default());
    det.observe(ts(5_000_000), good_metrics(100.0));
    let out = det.finalize_remaining(ts(1_000_000));
    assert!(
      out.is_empty(),
      "eos before earliest buffered ts should no-op"
    );
  }

  #[test]
  #[should_panic(expected = "target_interval must be > 0")]
  fn options_target_interval_zero_panics() {
    let _ = Options::new().with_target_interval(Duration::ZERO);
  }

  #[test]
  fn sharper_returns_ordering_greater() {
    assert!(sharper(2.0, 1.0));
    assert!(!sharper(1.0, 2.0));
    assert!(!sharper(1.0, 1.0));
    // NaN tolerant — not-greater either way.
    assert!(!sharper(f32::NAN, 1.0));
    assert!(!sharper(1.0, f32::NAN));
  }

  #[test]
  fn composite_weights_default_matches_spec() {
    let w = CompositeWeights::default();
    assert_eq!(w.sharpness(), 1.0);
    assert_eq!(w.sharpness_norm(), 1000.0);
    assert_eq!(w.noise(), 0.3);
    assert_eq!(w.noise_norm(), 20.0);
    assert_eq!(w.colorfulness(), 0.2);
    assert_eq!(w.colorfulness_norm(), 50.0);
    assert_eq!(w.clipping(), 0.5);
    assert_eq!(w.motion_blur(), 0.0);
  }

  #[test]
  fn composite_weights_builders_roundtrip() {
    let w = CompositeWeights::new()
      .with_sharpness(0.5, 500.0)
      .with_noise(0.1, 10.0)
      .with_colorfulness(0.4, 100.0)
      .with_clipping(0.25)
      .with_motion_blur(0.6);
    assert_eq!(w.sharpness(), 0.5);
    assert_eq!(w.sharpness_norm(), 500.0);
    assert_eq!(w.noise(), 0.1);
    assert_eq!(w.noise_norm(), 10.0);
    assert_eq!(w.colorfulness(), 0.4);
    assert_eq!(w.colorfulness_norm(), 100.0);
    assert_eq!(w.clipping(), 0.25);
    assert_eq!(w.motion_blur(), 0.6);
  }

  #[test]
  fn composite_weights_new_is_const_context_usable() {
    const W: CompositeWeights = CompositeWeights::new().with_motion_blur(0.5);
    assert_eq!(W.motion_blur(), 0.5);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn composite_weights_deserialize_clamps_invalid_norms_and_weights() {
    // Deserialization (via `#[serde(from = "CompositeWeightsRaw")]`)
    // routes every weight through `sanitise_weight` and every
    // normaliser through `sanitise_norm` so a serialized config
    // carrying NaN/Inf weights or zero/NaN/Inf norms cannot reach
    // composite_quality and silently corrupt the strict-pass argmax.
    // We exercise the conversion through `From<CompositeWeightsRaw>`
    // directly — that is the exact same code path serde-derive uses
    // after parsing the raw struct, and it doesn't pull in a
    // text-format crate for the test.
    let raw = CompositeWeightsRaw {
      sharpness: f32::NAN,              // invalid weight
      sharpness_norm: 0.0,              // invalid norm — zero
      noise: f32::INFINITY,             // invalid weight
      noise_norm: f32::NAN,             // invalid norm — NaN
      colorfulness: f32::NEG_INFINITY,  // invalid weight
      colorfulness_norm: f32::INFINITY, // invalid norm — +Inf
      clipping: f32::NAN,               // invalid weight
      motion_blur: f32::INFINITY,       // invalid weight
    };
    let w: CompositeWeights = raw.into();
    let defaults = CompositeWeights::new();
    // Norms fall back to spec defaults.
    assert_eq!(w.sharpness_norm(), defaults.sharpness_norm());
    assert_eq!(w.noise_norm(), defaults.noise_norm());
    assert_eq!(w.colorfulness_norm(), defaults.colorfulness_norm());
    // Weights clamp to 0 so each term contributes nothing.
    assert_eq!(w.sharpness(), 0.0);
    assert_eq!(w.noise(), 0.0);
    assert_eq!(w.colorfulness(), 0.0);
    assert_eq!(w.clipping(), 0.0);
    assert_eq!(w.motion_blur(), 0.0);

    // composite_quality on a normal frame stays finite — every term
    // collapses to 0 because weights are 0.
    let m = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_noise(5.0)
      .with_colorfulness(40.0)
      .with_clipping(0.5)
      .with_motion_blur(0.5);
    let q = composite_quality(&m, &w);
    assert!(q.is_finite());
    assert_eq!(q, 0.0);
  }

  #[cfg(feature = "serde")]
  #[test]
  fn composite_weights_deserialize_valid_norms_pass_through() {
    let raw = CompositeWeightsRaw {
      sharpness: 0.5,
      sharpness_norm: 250.0,
      noise: 0.1,
      noise_norm: 5.0,
      colorfulness: 0.4,
      colorfulness_norm: 200.0,
      clipping: 0.25,
      motion_blur: 0.6,
    };
    let w: CompositeWeights = raw.into();
    assert_eq!(w.sharpness_norm(), 250.0);
    assert_eq!(w.noise_norm(), 5.0);
    assert_eq!(w.colorfulness_norm(), 200.0);
  }

  #[test]
  fn composite_weights_paired_builders_are_const_context_usable() {
    // Compile-time evaluation through the sanitise_norm path:
    // - valid norms pass through
    // - invalid norms fall back to the spec defaults
    const VALID: CompositeWeights = CompositeWeights::new()
      .with_sharpness(0.5, 250.0)
      .with_noise(0.1, 5.0)
      .with_colorfulness(0.4, 200.0);
    assert_eq!(VALID.sharpness(), 0.5);
    assert_eq!(VALID.sharpness_norm(), 250.0);
    assert_eq!(VALID.noise(), 0.1);
    assert_eq!(VALID.noise_norm(), 5.0);
    assert_eq!(VALID.colorfulness(), 0.4);
    assert_eq!(VALID.colorfulness_norm(), 200.0);

    const CLAMPED: CompositeWeights = CompositeWeights::new()
      .with_sharpness(1.0, 0.0) // invalid: zero
      .with_noise(0.3, f32::NAN) // invalid: NaN
      .with_colorfulness(0.2, f32::INFINITY); // invalid: +Inf
    assert_eq!(CLAMPED.sharpness_norm(), 1000.0);
    assert_eq!(CLAMPED.noise_norm(), 20.0);
    assert_eq!(CLAMPED.colorfulness_norm(), 50.0);
    // Weights are stored verbatim regardless.
    assert_eq!(CLAMPED.sharpness(), 1.0);
    assert_eq!(CLAMPED.noise(), 0.3);
    assert_eq!(CLAMPED.colorfulness(), 0.2);
  }

  #[test]
  fn composite_argmax_picks_clean_over_sharper_noisy_under_defaults() {
    // Bucket with two strict-eligible frames:
    //   A: sharpness=2000, noise=15
    //   B: sharpness=1800, noise=3
    // Under default weights:
    //   q_A = 1.0·(2000/1000) - 0.3·(15/20) + 0 - 0 - 0  = 2.0 - 0.225 = 1.775
    //   q_B = 1.0·(1800/1000) - 0.3·( 3/20) + 0 - 0 - 0  = 1.8 - 0.045 = 1.755
    // A still wins by a hair (sharpness dominates), but bumping noise
    // weight should flip it.  Use a stronger noise weight here:
    let weights = CompositeWeights::new().with_noise(2.0, 20.0);
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);

    let a = FrameMetrics::new()
      .with_sharpness(2000.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(15.0);
    let b = FrameMetrics::new()
      .with_sharpness(1800.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(3.0);

    det.observe(ts(1_000_000), a);
    det.observe(ts(2_000_000), b);

    let out = det.finalize_shot(tr(0, 4_000_000));
    assert_eq!(out, vec![ts(2_000_000)]);
  }

  #[test]
  fn composite_argmax_collapses_to_sharpness_when_other_weights_zero() {
    // Zero out every non-sharpness weight → strict argmax must rank
    // by pure sharpness (mirrors legacy behaviour).
    let weights = CompositeWeights::new()
      .with_noise(0.0, 20.0)
      .with_colorfulness(0.0, 50.0)
      .with_clipping(0.0)
      .with_motion_blur(0.0);
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);

    // Same fixture as the existing `finalize_single_bucket_picks_sharpest`.
    det.observe(ts(0), good_metrics(100.0));
    det.observe(ts(500_000), good_metrics(500.0)); // sharpest
    det.observe(ts(1_500_000), good_metrics(200.0));

    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
  }

  #[test]
  fn adaptive_floor_recovers_strict_winner_in_low_detail_shot() {
    // 25 frames, all with sharpness in [20, 80] — well below the
    // absolute floor of 100. With adaptive_floor enabled, p25 ≈ 35,
    // so the strict gate passes any frame ≥ 35. The sharpest among
    // those becomes the strict winner instead of falling back.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    // 25 frames evenly spaced 0..25 seconds, sharpness ramping 20..80.
    for i in 0..25 {
      let s = 20.0 + (i as f32) * 2.5; // 20.0, 22.5, ..., 80.0
      det.observe(ts((i as i64) * 1_000_000), good_metrics(s));
    }
    // Composite-quality argmax with default weights → highest
    // composite wins. Since brightness/clipping/noise/etc are
    // identical, the highest-sharpness frame wins.
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out, vec![ts(24_000_000)]); // last frame, sharpness 80
  }

  #[test]
  fn adaptive_floor_disabled_falls_back_to_absolute_floor() {
    // 25 frames all below the absolute floor of 100. With adaptive
    // floor explicitly disabled, the strict gate rejects every frame
    // and we drop to fallback (pure sharpness). The result should
    // still be the sharpest frame — but via the fallback path.
    let weights = CompositeWeights::new().with_noise(10.0, 1.0); // huge noise penalty
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60))
      .with_adaptive_floor(false)
      .with_composite_weights(weights);
    let mut det = Detector::new(opts);
    let sharpest_with_noise = FrameMetrics::new()
      .with_sharpness(80.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_noise(1.0);
    for i in 0..24 {
      let s = 20.0 + (i as f32) * 2.5;
      det.observe(ts((i as i64) * 1_000_000), good_metrics(s));
    }
    det.observe(ts(24_000_000), sharpest_with_noise);
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out, vec![ts(24_000_000)]);
  }

  #[test]
  fn adaptive_floor_does_not_raise_floor_in_high_sharpness_shot() {
    // All frames at sharpness 500 — p25 = 500, well above the
    // absolute floor of 100. Effective floor must remain 100 (not
    // jump up to 500), so any frame with sharpness >= 100 still
    // passes.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    for i in 0..25 {
      det.observe(ts((i as i64) * 1_000_000), good_metrics(500.0));
    }
    let out = det.finalize_shot(tr(0, 30_000_000));
    assert_eq!(out.len(), 1, "exactly one winner expected");
  }

  #[test]
  fn adaptive_floor_uses_absolute_below_min_samples() {
    // Only 5 frames in the shot — below the default min_samples=20.
    // Adaptive floor must NOT activate; absolute floor applies.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_target_interval(Duration::from_secs(60));
    let mut det = Detector::new(opts);
    for i in 0..5 {
      det.observe(ts((i as i64) * 1_000_000), good_metrics(50.0)); // < 100
    }
    let out = det.finalize_shot(tr(0, 10_000_000));
    // No frame passes the absolute floor → fallback picks the only
    // candidate (all tied at 50.0).
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn motion_blur_gate_disabled_by_default_keeps_high_anisotropy_frame() {
    // Bucket with one strict-eligible frame whose motion_blur=0.9.
    // Gate off by default → frame passes strict, becomes the winner.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let m = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.9);
    det.observe(ts(500_000), m);
    let out = det.finalize_shot(tr(0, 2_000_000));
    assert_eq!(out, vec![ts(500_000)]);
  }

  #[test]
  fn motion_blur_gate_enabled_rejects_high_anisotropy_frame() {
    // Same fixture but with the gate on and a fresh, gate-passing
    // alternative. The high-anisotropy frame falls into fallback;
    // the low-anisotropy frame wins the strict path.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_motion_blur_gate(true)
      .with_max_motion_blur(0.75);
    let mut det = Detector::new(opts);
    let bad = FrameMetrics::new()
      .with_sharpness(800.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.9); // above the gate
    let good = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0)
      .with_motion_blur(0.1);
    det.observe(ts(500_000), bad);
    det.observe(ts(1_500_000), good);
    let out = det.finalize_shot(tr(0, 2_000_000));
    // Strict path: `good` wins (bad rejected by motion-blur gate).
    assert_eq!(out, vec![ts(1_500_000)]);
  }

  // ---- Normalisation / range guard regressions -----------------------------

  #[test]
  fn composite_weights_invalid_norms_clamp_to_default() {
    // All flavours of invalid (zero, negative, NaN, +inf, -inf) must
    // fall back to the spec defaults so composite_quality never sees
    // an Inf/NaN-producing divisor.
    let defaults = CompositeWeights::new();
    for &bad in &[
      0.0f32,
      -1.0,
      -0.0,
      f32::NAN,
      f32::INFINITY,
      f32::NEG_INFINITY,
    ] {
      let w = CompositeWeights::new()
        .with_sharpness(1.0, bad)
        .with_noise(0.3, bad)
        .with_colorfulness(0.2, bad);
      assert_eq!(
        w.sharpness_norm(),
        defaults.sharpness_norm(),
        "sharpness_norm should clamp invalid {bad:?} to default"
      );
      assert_eq!(
        w.noise_norm(),
        defaults.noise_norm(),
        "noise_norm should clamp invalid {bad:?} to default"
      );
      assert_eq!(
        w.colorfulness_norm(),
        defaults.colorfulness_norm(),
        "colorfulness_norm should clamp invalid {bad:?} to default"
      );
      // The weight itself is stored verbatim — clamp is on `norm` only.
      assert_eq!(w.sharpness(), 1.0);
      assert_eq!(w.noise(), 0.3);
      assert_eq!(w.colorfulness(), 0.2);
    }
  }

  #[test]
  fn composite_weights_valid_norms_pass_through() {
    let w = CompositeWeights::new()
      .with_sharpness(0.5, 250.0)
      .with_noise(0.1, 5.0)
      .with_colorfulness(0.4, 200.0);
    assert_eq!(w.sharpness_norm(), 250.0);
    assert_eq!(w.noise_norm(), 5.0);
    assert_eq!(w.colorfulness_norm(), 200.0);
  }

  #[test]
  fn composite_quality_with_clamped_norms_stays_finite() {
    // Belt-and-braces: even if a caller chains every kind of invalid
    // norm onto the weights, composite_quality must produce a finite
    // result on a normal frame so the strict-pass argmax keeps
    // ranking deterministically.
    let weights = CompositeWeights::new()
      .with_sharpness(1.0, f32::NAN)
      .with_noise(0.3, 0.0)
      .with_colorfulness(0.2, f32::INFINITY);
    let m = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_noise(5.0)
      .with_colorfulness(40.0)
      .with_clipping(0.0)
      .with_motion_blur(0.0);
    let q = composite_quality(&m, &weights);
    assert!(
      q.is_finite(),
      "composite_quality must stay finite under clamped invalid norms; got {q}"
    );
  }

  #[test]
  fn composite_weights_invalid_weights_clamp_to_zero() {
    // Every flavour of non-finite weight on every builder must clamp
    // to 0.0, so the term contributes nothing rather than poisoning
    // composite_quality with `Inf`/`NaN`.
    for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
      let w = CompositeWeights::new()
        .with_sharpness(bad, 1000.0)
        .with_noise(bad, 20.0)
        .with_colorfulness(bad, 50.0)
        .with_clipping(bad)
        .with_motion_blur(bad);
      assert_eq!(w.sharpness(), 0.0, "sharpness should clamp invalid {bad}");
      assert_eq!(w.noise(), 0.0, "noise should clamp invalid {bad}");
      assert_eq!(
        w.colorfulness(),
        0.0,
        "colorfulness should clamp invalid {bad}"
      );
      assert_eq!(w.clipping(), 0.0, "clipping should clamp invalid {bad}");
      assert_eq!(
        w.motion_blur(),
        0.0,
        "motion_blur should clamp invalid {bad}"
      );
    }
  }

  #[test]
  fn composite_weights_finite_negative_weight_passes_through() {
    // Negative weights are well-defined (a user can invert a term's
    // sense deliberately). Only non-finite weights are filtered.
    let w = CompositeWeights::new()
      .with_sharpness(-1.0, 1000.0)
      .with_clipping(-0.5);
    assert_eq!(w.sharpness(), -1.0);
    assert_eq!(w.clipping(), -0.5);
  }

  #[test]
  fn composite_quality_stays_finite_under_invalid_weights() {
    // End-to-end guard: non-finite weights on every term + a normal
    // frame must still produce a finite composite. Catches any future
    // builder that forgets to route a weight through sanitise_weight.
    let weights = CompositeWeights::new()
      .with_sharpness(f32::NAN, 1000.0)
      .with_noise(f32::INFINITY, 20.0)
      .with_colorfulness(f32::NEG_INFINITY, 50.0)
      .with_clipping(f32::NAN)
      .with_motion_blur(f32::INFINITY);
    let m = FrameMetrics::new()
      .with_sharpness(500.0)
      .with_noise(5.0)
      .with_colorfulness(40.0)
      .with_clipping(0.5)
      .with_motion_blur(0.5);
    let q = composite_quality(&m, &weights);
    assert!(
      q.is_finite(),
      "composite_quality must stay finite under clamped invalid weights; got {q}"
    );
    // Every term clamped to weight=0 → q == 0.
    assert_eq!(q, 0.0);
  }

  #[test]
  fn strict_argmax_skips_non_finite_composite() {
    // FrameMetrics setters accept any f32, including non-finite values
    // — a corrupt detector output or a malformed caller could push
    // NaN/Inf through. The strict-pass argmax must NOT lock onto a
    // non-finite composite (which would prevent later finite candidates
    // from unseating it). Two frames: the first has a NaN sharpness;
    // the second is a normal in-bucket frame. The strict winner must
    // be the second.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let poisoned = FrameMetrics::new()
      .with_sharpness(f32::NAN) // poisons composite_quality via division-by-norm
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0);
    let normal = good_metrics(500.0);
    det.observe(ts(500_000), poisoned);
    det.observe(ts(1_500_000), normal);
    let out = det.finalize_shot(tr(0, 2_000_000));
    // The non-finite-composite frame is skipped from strict; the
    // normal frame wins.
    assert_eq!(out, vec![ts(1_500_000)]);
  }

  #[test]
  fn strict_argmax_falls_back_when_only_candidate_is_non_finite() {
    // Single frame with a NaN sharpness fails the strict path → drops
    // into the fallback (raw-sharpness) path, which still selects it
    // as the "least bad" candidate. Verifies that the guard doesn't
    // accidentally swallow the only candidate.
    let opts = Options::default().with_margin_ratio(0.0);
    let mut det = Detector::new(opts);
    let m = FrameMetrics::new()
      .with_sharpness(f32::NAN)
      .with_brightness(128.0)
      .with_luma_variance(200.0)
      .with_saturation_variance(100.0)
      .with_clipping(0.0);
    det.observe(ts(500_000), m);
    let out = det.finalize_shot(tr(0, 2_000_000));
    // Fallback path emits one timestamp.
    assert_eq!(out, vec![ts(500_000)]);
  }

  #[test]
  fn adaptive_floor_min_samples_zero_with_empty_shot_uses_absolute_floor() {
    // min_samples = 0 + empty in-range shot would previously index an
    // empty Vec and panic. The guard must fall back to the absolute
    // floor and return cleanly.
    let opts = Options::default()
      .with_margin_ratio(0.0)
      .with_adaptive_floor_min_samples(0);
    let mut det = Detector::new(opts);
    // No observations — finalize a non-empty range and expect no emits
    // (and no panic).
    let out = det.finalize_shot(tr(0, 4_000_000));
    assert!(out.is_empty(), "empty shot must yield no keyframes");
  }
}
