//! Content-change scene detection via HSV-space deltas and optional Canny edges.
//!
//! This module implements [`Detector`], a port of PySceneDetect's
//! `detect-content`. For each consecutive frame pair it computes up to four
//! per-channel L1 differences in HSV color space (plus optionally a Canny
//! edge map), combines them into a weighted **`frame_score`**, and emits a
//! cut when the score exceeds [`Options::threshold`].
//!
//! # Pipeline
//!
//! For each frame:
//!
//! 1. **Obtain HSV planes.** Either supplied directly (`process_hsv`),
//!    converted from a packed BGR frame (`process_bgr`), or — in luma-only
//!    mode — taken as the Y plane alone (`process_luma`).
//! 2. **Optionally compute edges** on the V plane via Canny + morphological
//!    dilation. Skipped when `weights.delta_edges == 0.0`.
//! 3. **Compute four component deltas** against the previous frame's
//!    corresponding planes:
//!    - `delta_hue`, `delta_sat`, `delta_lum` — mean(|curr − prev|).
//!    - `delta_edges` — same, but over the dilated binary edge maps.
//! 4. **Combine into `frame_score`** as `Σ(component × weight) / Σ|weight|`.
//! 5. **Apply threshold + min-duration gate** via the selected [`FilterMode`].
//!
//! # Entry points
//!
//! | Method | Input | Notes |
//! |---|---|---|
//! | [`Detector::process_luma`] | [`LumaFrame`] | Hue / Saturation weights ignored (we have no chroma). Use when weights are luma-only. |
//! | [`Detector::process_bgr`] | [`RgbFrame`] | Full pipeline. Byte layout is B,G,R per pixel. |
//! | [`Detector::process_hsv`] | [`HsvFrame`] | Skip HSV conversion — assumes OpenCV's 8-bit encoding (H in `[0, 179]`). |
//!
//! # Filter modes
//!
//! [`FilterMode::Suppress`] — emit a cut when score ≥ threshold and at
//! least `min_duration` has elapsed since the previous cut.
//!
//! [`FilterMode::Merge`] (default, matches Python) — collapse rapid
//! consecutive above-threshold frames into a single cut emitted after the
//! signal has stayed below threshold for `min_duration`. See [`Options::initial_cut`]
//! for the first-cut behavior.
//!
//! # Attribution
//!
//! Ported from PySceneDetect's `detect-content` (BSD 3-Clause). HSV
//! conversion matches OpenCV's `cv2.COLOR_BGR2HSV` semantics; Canny +
//! dilate follow the same shape as `cv2.Canny` + `cv2.dilate`.

use core::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::frame::{HsvFrame, LumaFrame, RgbFrame, Timebase, Timestamp};

/// Default weights for the four score components. Matches PySceneDetect's
/// `DEFAULT_COMPONENT_WEIGHTS`: hue, saturation, and luma equally weighted;
/// edges off.
pub const DEFAULT_WEIGHTS: Components = Components::new(1.0, 1.0, 1.0, 0.0);

/// Weights that ignore color and score only on luma change. Matches
/// PySceneDetect's `LUMA_ONLY_WEIGHTS`.
pub const LUMA_ONLY_WEIGHTS: Components = Components::new(0.0, 0.0, 1.0, 0.0);

/// The four components that combine into a content-change score.
///
/// Each weight applies to the corresponding L1 difference between
/// consecutive frames. Use signed weights to down-weight a channel or to
/// combine in unusual ways; the score normalization divides by the sum of
/// absolute weights.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Components {
  delta_hue: f64,
  delta_sat: f64,
  delta_lum: f64,
  delta_edges: f64,
}

impl Components {
  /// Creates a new [`Components`] with the given weights.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(delta_hue: f64, delta_sat: f64, delta_lum: f64, delta_edges: f64) -> Self {
    Self {
      delta_hue,
      delta_sat,
      delta_lum,
      delta_edges,
    }
  }

  /// Weight for mean |ΔH| (hue channel, `[0, 179]` in OpenCV's encoding).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn delta_hue(&self) -> f64 {
    self.delta_hue
  }

  /// Sets the hue-delta weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_delta_hue(mut self, val: f64) -> Self {
    self.delta_hue = val;
    self
  }

  /// Sets the hue-delta weight in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_delta_hue(&mut self, val: f64) -> &mut Self {
    self.delta_hue = val;
    self
  }

  /// Weight for mean |ΔS| (saturation channel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn delta_sat(&self) -> f64 {
    self.delta_sat
  }

  /// Sets the saturation-delta weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_delta_sat(mut self, val: f64) -> Self {
    self.delta_sat = val;
    self
  }

  /// Sets the saturation-delta weight in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_delta_sat(&mut self, val: f64) -> &mut Self {
    self.delta_sat = val;
    self
  }

  /// Weight for mean |ΔV| (value / luma channel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn delta_lum(&self) -> f64 {
    self.delta_lum
  }

  /// Sets the luma-delta weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_delta_lum(mut self, val: f64) -> Self {
    self.delta_lum = val;
    self
  }

  /// Sets the luma-delta weight in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_delta_lum(&mut self, val: f64) -> &mut Self {
    self.delta_lum = val;
    self
  }

  /// Weight for mean |ΔE| over the dilated Canny edge map on V.
  /// Non-zero enables edge detection (expensive); zero skips it.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn delta_edges(&self) -> f64 {
    self.delta_edges
  }

  /// Sets the edge-delta weight. Non-zero enables Canny edge detection.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_delta_edges(mut self, val: f64) -> Self {
    self.delta_edges = val;
    self
  }

  /// Sets the edge-delta weight in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_delta_edges(&mut self, val: f64) -> &mut Self {
    self.delta_edges = val;
    self
  }

  /// Returns the sum of absolute weights. Used for score normalization.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn sum_abs(&self) -> f64 {
    self.delta_hue.abs() + self.delta_sat.abs() + self.delta_lum.abs() + self.delta_edges.abs()
  }
}

impl Default for Components {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    DEFAULT_WEIGHTS
  }
}

/// How the detector gates cut emission against [`Options::min_duration`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum FilterMode {
  /// Emit a cut only when the score ≥ threshold **and** at least
  /// `min_duration` has elapsed since the previous above-threshold frame.
  /// Cuts within the gate are silently dropped.
  Suppress,
  /// Collapse rapid consecutive above-threshold frames into a single cut.
  /// Default — matches PySceneDetect.
  #[default]
  Merge,
}

/// Error returned by [`Detector::try_new`] when the provided [`Options`] are
/// inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// All component weights are zero — the score would always be `NaN`
  /// (0/0) or always zero. Set at least one weight non-zero.
  #[error("all component weights are zero")]
  ZeroWeights,
  /// `kernel_size` was smaller than 3 or even. Must be an odd integer ≥ 3.
  #[error("kernel_size ({0}) must be an odd integer >= 3")]
  InvalidKernelSize(u32),
}

/// Options for the content-change scene detector. See the
/// [module docs](crate::content) for the full algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  threshold: f64,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  min_duration: Duration,
  weights: Components,
  filter_mode: FilterMode,
  /// Edge-dilation kernel size. `None` = auto-compute from frame dimensions.
  #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
  kernel_size: Option<u32>,
  initial_cut: bool,
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl Options {
  /// Creates a new `Options` with default values.
  ///
  /// Defaults: `threshold = 27.0`, `min_duration = 1 s`, weights =
  /// [`DEFAULT_WEIGHTS`], filter mode = [`FilterMode::Merge`],
  /// auto kernel size, `initial_cut = true`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold: 27.0,
      min_duration: Duration::from_secs(1),
      weights: DEFAULT_WEIGHTS,
      filter_mode: FilterMode::Merge,
      kernel_size: None,
      initial_cut: true,
    }
  }

  /// Returns the score threshold required to trigger a cut.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }

  /// Sets the score threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold(mut self, val: f64) -> Self {
    self.threshold = val;
    self
  }

  /// Sets the score threshold in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threshold(&mut self, val: f64) -> &mut Self {
    self.threshold = val;
    self
  }

  /// Returns the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_duration(&self) -> Duration {
    self.min_duration
  }

  /// Sets the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_duration(mut self, val: Duration) -> Self {
    self.min_duration = val;
    self
  }

  /// Sets the minimum scene duration in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_duration(&mut self, val: Duration) -> &mut Self {
    self.min_duration = val;
    self
  }

  /// Set minimum scene length as a number of frames at a given frame rate.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_frames(mut self, frames: u32, fps: Timebase) -> Self {
    self.min_duration = fps.frames_to_duration(frames);
    self
  }

  /// In-place form of [`Self::with_min_frames`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_frames(&mut self, frames: u32, fps: Timebase) -> &mut Self {
    self.min_duration = fps.frames_to_duration(frames);
    self
  }

  /// Returns the per-component weights.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn weights(&self) -> Components {
    self.weights
  }

  /// Sets the per-component weights.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_weights(mut self, val: Components) -> Self {
    self.weights = val;
    self
  }

  /// Sets the per-component weights in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_weights(&mut self, val: Components) -> &mut Self {
    self.weights = val;
    self
  }

  /// Returns the filter mode.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn filter_mode(&self) -> FilterMode {
    self.filter_mode
  }

  /// Sets the filter mode.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_filter_mode(mut self, val: FilterMode) -> Self {
    self.filter_mode = val;
    self
  }

  /// Sets the filter mode in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_filter_mode(&mut self, val: FilterMode) -> &mut Self {
    self.filter_mode = val;
    self
  }

  /// Returns the edge-dilation kernel size, or `None` for auto-compute.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kernel_size(&self) -> Option<u32> {
    self.kernel_size
  }

  /// Sets the kernel size (must be odd and ≥ 3 at detector construction time).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_kernel_size(mut self, val: Option<u32>) -> Self {
    self.kernel_size = val;
    self
  }

  /// Sets the kernel size in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_kernel_size(&mut self, val: Option<u32>) -> &mut Self {
    self.kernel_size = val;
    self
  }

  /// Whether the first above-threshold transition is allowed to emit a cut
  /// immediately, bypassing the warmup window that MERGE/SUPPRESS would
  /// otherwise enforce at stream start.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn initial_cut(&self) -> bool {
    self.initial_cut
  }

  /// Sets `initial_cut`.
  ///
  /// - `true` (default): the first real cut fires as soon as the score
  ///   crosses the threshold.
  /// - `false`: matches PySceneDetect — suppresses cuts until the stream
  ///   has actually run for at least `min_duration`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_initial_cut(mut self, val: bool) -> Self {
    self.initial_cut = val;
    self
  }

  /// Sets `initial_cut` in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_initial_cut(&mut self, val: bool) -> &mut Self {
    self.initial_cut = val;
    self
  }
}

/// Content-change scene detector.
///
/// See [module documentation](crate::content) for the algorithm.
///
/// Per-frame scratch buffers (HSV history, scratch planes, optional edge
/// scratch) are allocated lazily on the first frame — once the input
/// resolution is known. A dimension change triggers a reallocation, so
/// streams that change resolution mid-stream still work, though without
/// zero-alloc steady-state.
#[derive(Debug, Clone)]
pub struct Detector {
  options: Options,
  /// Sum of absolute weights, precomputed once.
  sum_abs_weights: f64,
  /// Whether we should compute the edge component at all.
  edges_enabled: bool,
  // Stream state
  has_previous: bool,
  last_score: Option<f64>,
  last_components: Option<Components>,
  // Flash filter state
  last_above: Option<Timestamp>,
  merge_enabled: bool,
  merge_triggered: bool,
  merge_start: Option<Timestamp>,
  // Per-frame scratch (lazy-allocated)
  width: u32,
  height: u32,
  kernel: u32,
  prev_h: Vec<u8>,
  prev_s: Vec<u8>,
  prev_v: Vec<u8>,
  prev_edges: Vec<u8>,
  cur_h: Vec<u8>,
  cur_s: Vec<u8>,
  cur_v: Vec<u8>,
  cur_edges: Vec<u8>,
  // Canny scratch
  sobel_mag: Vec<i32>,
  sobel_dir: Vec<u8>,
  nms_out: Vec<u8>,
  dilate_tmp: Vec<u8>,
}

impl Detector {
  /// Creates a new detector with the given options.
  ///
  /// # Panics
  ///
  /// Panics if the options are invalid — see [`Error`].
  pub fn new(options: Options) -> Self {
    Self::try_new(options).expect("invalid content::Options")
  }

  /// Creates a new detector with the given options, returning [`Error`] on
  /// invalid configuration.
  pub fn try_new(options: Options) -> Result<Self, Error> {
    let sum = options.weights.sum_abs();
    if sum == 0.0 {
      return Err(Error::ZeroWeights);
    }
    if let Some(k) = options.kernel_size {
      if k < 3 || k % 2 == 0 {
        return Err(Error::InvalidKernelSize(k));
      }
    }
    let edges_enabled = options.weights.delta_edges != 0.0;

    Ok(Self {
      options,
      sum_abs_weights: sum,
      edges_enabled,
      has_previous: false,
      last_score: None,
      last_components: None,
      last_above: None,
      merge_enabled: false,
      merge_triggered: false,
      merge_start: None,
      width: 0,
      height: 0,
      kernel: 0,
      prev_h: Vec::new(),
      prev_s: Vec::new(),
      prev_v: Vec::new(),
      prev_edges: Vec::new(),
      cur_h: Vec::new(),
      cur_s: Vec::new(),
      cur_v: Vec::new(),
      cur_edges: Vec::new(),
      sobel_mag: Vec::new(),
      sobel_dir: Vec::new(),
      nms_out: Vec::new(),
      dilate_tmp: Vec::new(),
    })
  }

  /// Returns a reference to the options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// Returns the computed score for the most recently processed frame, or
  /// `None` if fewer than two frames have been processed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_score(&self) -> Option<f64> {
    self.last_score
  }

  /// Returns the last frame's per-component deltas (unweighted), or `None`
  /// if fewer than two frames have been processed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_components(&self) -> Option<Components> {
    self.last_components
  }

  /// Resets streaming state so this detector instance can be reused.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.has_previous = false;
    self.last_score = None;
    self.last_components = None;
    self.last_above = None;
    self.merge_enabled = false;
    self.merge_triggered = false;
    self.merge_start = None;
  }

  /// Processes a luma-only frame. Hue and saturation components are treated
  /// as zero (no chroma available); only `delta_lum` and `delta_edges`
  /// contribute to the score.
  pub fn process_luma(&mut self, frame: LumaFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.ensure_buffers(frame.width(), frame.height());
    copy_plane(
      &mut self.cur_v,
      frame.data(),
      frame.width(),
      frame.height(),
      frame.stride(),
    );
    // Zero hue & saturation — they won't affect the score if weights are zero
    // (as in luma-only), and contribute a constant 0 delta otherwise.
    for slot in self.cur_h.iter_mut() {
      *slot = 0;
    }
    for slot in self.cur_s.iter_mut() {
      *slot = 0;
    }

    self.process_inner(ts)
  }

  /// Processes a packed 24-bit BGR frame. Converts to HSV internally.
  pub fn process_bgr(&mut self, frame: RgbFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.ensure_buffers(frame.width(), frame.height());
    bgr_to_hsv_planes(
      &mut self.cur_h,
      &mut self.cur_s,
      &mut self.cur_v,
      frame.data(),
      frame.width(),
      frame.height(),
      frame.stride(),
    );
    self.process_inner(ts)
  }

  /// Processes an already-converted HSV frame. Assumes OpenCV's 8-bit HSV
  /// encoding (H in `[0, 179]`).
  pub fn process_hsv(&mut self, frame: HsvFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.ensure_buffers(frame.width(), frame.height());
    copy_plane(
      &mut self.cur_h,
      frame.hue(),
      frame.width(),
      frame.height(),
      frame.stride(),
    );
    copy_plane(
      &mut self.cur_s,
      frame.saturation(),
      frame.width(),
      frame.height(),
      frame.stride(),
    );
    copy_plane(
      &mut self.cur_v,
      frame.value(),
      frame.width(),
      frame.height(),
      frame.stride(),
    );
    self.process_inner(ts)
  }

  /// Shared logic after planes are filled into `cur_h/s/v`.
  fn process_inner(&mut self, ts: Timestamp) -> Option<Timestamp> {
    let n = (self.width as usize) * (self.height as usize);

    // Edges (before computing score, since we need them before swapping).
    if self.edges_enabled {
      self.compute_edges();
    }

    // Compute components and score only after the first frame.
    let mut cut: Option<Timestamp> = None;
    if self.has_previous {
      let components = Components::new(
        mean_abs_diff(&self.cur_h, &self.prev_h, n),
        mean_abs_diff(&self.cur_s, &self.prev_s, n),
        mean_abs_diff(&self.cur_v, &self.prev_v, n),
        if self.edges_enabled {
          mean_abs_diff(&self.cur_edges, &self.prev_edges, n)
        } else {
          0.0
        },
      );
      let w = self.options.weights;
      let score = (components.delta_hue() * w.delta_hue()
        + components.delta_sat() * w.delta_sat()
        + components.delta_lum() * w.delta_lum()
        + components.delta_edges() * w.delta_edges())
        / self.sum_abs_weights;

      self.last_score = Some(score);
      self.last_components = Some(components);

      let above = score >= self.options.threshold;
      cut = self.flash_filter(ts, above);
    }

    // Swap current → previous.
    core::mem::swap(&mut self.prev_h, &mut self.cur_h);
    core::mem::swap(&mut self.prev_s, &mut self.cur_s);
    core::mem::swap(&mut self.prev_v, &mut self.cur_v);
    if self.edges_enabled {
      core::mem::swap(&mut self.prev_edges, &mut self.cur_edges);
    }
    self.has_previous = true;

    cut
  }

  /// Full Canny + dilate pipeline on the current V plane, writing the dilated
  /// edge map into `self.cur_edges`.
  ///
  /// Canny thresholds are derived from the median of the V plane
  /// (`sigma = 1/3`) to mirror the auto-threshold pattern PySceneDetect
  /// uses with `cv2.Canny`.
  fn compute_edges(&mut self) {
    // Pre-grab disjoint-field borrows so the sub-passes can run without the
    // borrow checker needing to reason about re-borrowing `self`.
    let input = &self.cur_v;
    let sobel_mag = &mut self.sobel_mag;
    let sobel_dir = &mut self.sobel_dir;
    let nms_out = &mut self.nms_out;
    let tmp = &mut self.dilate_tmp;
    let out = &mut self.cur_edges;
    let width = self.width;
    let height = self.height;
    let kernel = self.kernel;

    let median = median_u8(input);
    let sigma = 1.0_f32 / 3.0;
    let low = ((1.0 - sigma) * median as f32).max(0.0) as u8;
    let high = ((1.0 + sigma) * median as f32).min(255.0) as u8;

    sobel(input, sobel_mag, sobel_dir, width, height);
    non_max_suppress(sobel_mag, sobel_dir, nms_out, width, height);
    hysteresis(nms_out, sobel_mag, low, high, width, height);
    dilate(nms_out, out, tmp, width, height, kernel);
  }

  /// Apply MERGE or SUPPRESS gating.
  fn flash_filter(&mut self, ts: Timestamp, above: bool) -> Option<Timestamp> {
    // Seed `last_above` on first call.
    if self.last_above.is_none() {
      self.last_above = Some(virtual_seed(ts, &self.options));
    }

    let last_above_ts = self.last_above.expect("seeded above");
    let min_length_met = ts
      .duration_since(&last_above_ts)
      .is_some_and(|d| d >= self.options.min_duration);

    match self.options.filter_mode {
      FilterMode::Suppress => {
        if !above || !min_length_met {
          if above {
            // Track presence (Python behavior) — SUPPRESS updates last_above
            // only when it emits, but we need it for min_length tracking.
            // Match Python: update only on emission.
          }
          // Did NOT emit.
          None
        } else {
          self.last_above = Some(ts);
          Some(ts)
        }
      }
      FilterMode::Merge => self.filter_merge(ts, above, min_length_met),
    }
  }

  fn filter_merge(
    &mut self,
    ts: Timestamp,
    above: bool,
    min_length_met: bool,
  ) -> Option<Timestamp> {
    // Always advance `last_above` when above.
    if above {
      self.last_above = Some(ts);
    }

    if self.merge_triggered {
      // Currently holding cuts back; check if we can release one.
      let merge_start = self.merge_start.expect("triggered implies start");
      let last_above = self.last_above.expect("seeded above");
      let num_merged = last_above
        .duration_since(&merge_start)
        .unwrap_or(Duration::ZERO);
      if min_length_met && !above && num_merged >= self.options.min_duration {
        self.merge_triggered = false;
        return self.last_above;
      }
      return None;
    }
    if !above {
      return None;
    }
    if min_length_met {
      // Meets min-length: emit the cut and arm the merge for subsequent
      // rapid-cut suppression.
      self.merge_enabled = true;
      return Some(ts);
    }
    // Not min-length; trigger merge only after at least one cut was emitted.
    if self.merge_enabled {
      self.merge_triggered = true;
      self.merge_start = Some(ts);
    }
    None
  }

  /// Ensure all per-frame buffers are sized for the current frame. Reallocs
  /// on first frame or dimension change; no-op otherwise.
  fn ensure_buffers(&mut self, width: u32, height: u32) {
    if self.width == width && self.height == height {
      return;
    }
    self.width = width;
    self.height = height;
    self.kernel = self
      .options
      .kernel_size
      .unwrap_or_else(|| auto_kernel_size(width, height));

    let n = (width as usize) * (height as usize);
    for v in [
      &mut self.prev_h,
      &mut self.prev_s,
      &mut self.prev_v,
      &mut self.cur_h,
      &mut self.cur_s,
      &mut self.cur_v,
    ] {
      v.clear();
      v.resize(n, 0);
    }
    if self.edges_enabled {
      for v in [
        &mut self.prev_edges,
        &mut self.cur_edges,
        &mut self.nms_out,
        &mut self.dilate_tmp,
      ] {
        v.clear();
        v.resize(n, 0);
      }
      self.sobel_mag.clear();
      self.sobel_mag.resize(n, 0);
      self.sobel_dir.clear();
      self.sobel_dir.resize(n, 0);
    }
    // Re-seed the flash filter on dimension change (new stream semantics).
    self.last_above = None;
    self.merge_enabled = false;
    self.merge_triggered = false;
    self.merge_start = None;
    self.has_previous = false;
  }
}

/// Seeds the flash filter's `last_above` to either the current timestamp
/// (Python-compat suppressing an early cut) or to a virtual past point
/// (`ts - min_duration`, so the first above-threshold frame passes the gate).
fn virtual_seed(ts: Timestamp, options: &Options) -> Timestamp {
  if options.initial_cut {
    ts.saturating_sub_duration(options.min_duration)
  } else {
    ts
  }
}

// -----------------------------------------------------------------------------
// Per-pixel helpers
// -----------------------------------------------------------------------------

/// Copies a strided plane into a packed `dst` of length `width * height`.
fn copy_plane(dst: &mut [u8], src: &[u8], width: u32, height: u32, stride: u32) {
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  for y in 0..h {
    let dst_row = &mut dst[y * w..(y + 1) * w];
    let src_row = &src[y * s..y * s + w];
    dst_row.copy_from_slice(src_row);
  }
}

/// Mean of the absolute per-pixel difference over `n` values.
fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  debug_assert!(a.len() >= n && b.len() >= n);
  let mut sum: u64 = 0;
  for i in 0..n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
  }
  if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

// -----------------------------------------------------------------------------
// BGR → HSV (OpenCV-compatible 8-bit encoding; H in [0, 179])
// -----------------------------------------------------------------------------

/// Converts a packed 24-bit BGR frame into three planar HSV buffers matching
/// OpenCV's `cv2.COLOR_BGR2HSV` semantics.
fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
) {
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  for y in 0..h {
    let row = &src[y * s..y * s + w * 3];
    let dst_off = y * w;
    for x in 0..w {
      let b = row[x * 3] as f32;
      let g = row[x * 3 + 1] as f32;
      let r = row[x * 3 + 2] as f32;
      let (hue, sat, val) = bgr_to_hsv_pixel(b, g, r);
      h_out[dst_off + x] = hue;
      s_out[dst_off + x] = sat;
      v_out[dst_off + x] = val;
    }
  }
}

#[inline]
fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
  let v = b.max(g).max(r);
  let min = b.min(g).min(r);
  let delta = v - min;
  let s = if v == 0.0 { 0.0 } else { 255.0 * delta / v };
  let hue = if delta == 0.0 {
    0.0
  } else if v == r {
    let h = 60.0 * (g - b) / delta;
    if h < 0.0 { h + 360.0 } else { h }
  } else if v == g {
    60.0 * (b - r) / delta + 120.0
  } else {
    60.0 * (r - g) / delta + 240.0
  };
  let h8 = (hue * 0.5).round().clamp(0.0, 179.0) as u8;
  (
    h8,
    s.round().clamp(0.0, 255.0) as u8,
    v.round().clamp(0.0, 255.0) as u8,
  )
}

// -----------------------------------------------------------------------------
// Canny edge detection + morphological dilation (square kernel)
// -----------------------------------------------------------------------------

/// Auto kernel-size heuristic matching PySceneDetect: `4 + round(sqrt(w*h)/192)`,
/// bumped to odd.
fn auto_kernel_size(width: u32, height: u32) -> u32 {
  let d = ((width as f64 * height as f64).sqrt() / 192.0).round() as u32;
  let mut k = 4 + d;
  if k % 2 == 0 {
    k += 1;
  }
  k.max(3)
}

/// Median of a `[u8]` via histogram — O(N) and parallel-unrollable.
fn median_u8(buf: &[u8]) -> u8 {
  let mut hist = [0u32; 256];
  for &v in buf {
    hist[v as usize] += 1;
  }
  let half = buf.len() as u32 / 2;
  let mut cum = 0u32;
  for (i, &c) in hist.iter().enumerate() {
    cum += c;
    if cum > half {
      return i as u8;
    }
  }
  255
}

/// 3×3 Sobel: computes magnitude (`|Gx| + |Gy|`, L1) and a quantized
/// gradient direction (0=horizontal, 1=45°, 2=vertical, 3=135°).
/// Border pixels get magnitude 0.
fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], width: u32, height: u32) {
  let w = width as usize;
  let h = height as usize;
  for v in mag.iter_mut() {
    *v = 0;
  }
  for v in dir.iter_mut() {
    *v = 0;
  }
  for y in 1..h.saturating_sub(1) {
    for x in 1..w.saturating_sub(1) {
      let i = |yy: usize, xx: usize| input[yy * w + xx] as i32;
      // Gx: [-1 0 1; -2 0 2; -1 0 1]
      let gx = -i(y - 1, x - 1) - 2 * i(y, x - 1) - i(y + 1, x - 1)
        + i(y - 1, x + 1)
        + 2 * i(y, x + 1)
        + i(y + 1, x + 1);
      // Gy: [-1 -2 -1; 0 0 0; 1 2 1]
      let gy = -i(y - 1, x - 1) - 2 * i(y - 1, x) - i(y - 1, x + 1)
        + i(y + 1, x - 1)
        + 2 * i(y + 1, x)
        + i(y + 1, x + 1);
      let m = gx.abs() + gy.abs();
      let idx = y * w + x;
      mag[idx] = m;
      // Quantize direction: angle = atan2(gy, gx), quantize to 4 bins.
      let ax = gx.abs();
      let ay = gy.abs();
      // Compare gy/gx ratio against tan(22.5°)≈0.414 and tan(67.5°)≈2.414.
      // ay / ax < 0.414 → horizontal (0)
      // 0.414 ≤ ay/ax < 2.414 → diagonal — sign determines 45° (1) vs 135° (3)
      // ay/ax ≥ 2.414 → vertical (2)
      let d: u8 = if ay * 1000 < ax * 414 {
        0
      } else if ay * 1000 > ax * 2414 {
        2
      } else if gx.signum() == gy.signum() {
        1
      } else {
        3
      };
      dir[idx] = d;
    }
  }
}

/// Non-maximum suppression along gradient direction. Pixels that aren't a
/// local max in the gradient direction are zeroed; survivors retain their
/// magnitude (clamped to u8 for downstream hysteresis, with true magnitude
/// in `mag` preserved for the high-threshold check).
fn non_max_suppress(mag: &[i32], dir: &[u8], out: &mut [u8], width: u32, height: u32) {
  let w = width as usize;
  let h = height as usize;
  for v in out.iter_mut() {
    *v = 0;
  }
  for y in 1..h.saturating_sub(1) {
    for x in 1..w.saturating_sub(1) {
      let idx = y * w + x;
      let m = mag[idx];
      if m == 0 {
        continue;
      }
      let (dx, dy): (isize, isize) = match dir[idx] {
        0 => (1, 0),  // horizontal
        1 => (1, 1),  // 45°
        2 => (0, 1),  // vertical
        _ => (1, -1), // 135°
      };
      let a = mag[((y as isize + dy) as usize) * w + (x as isize + dx) as usize];
      let b = mag[((y as isize - dy) as usize) * w + (x as isize - dx) as usize];
      if m >= a && m >= b {
        // Clamp magnitude to u8 for output.
        out[idx] = m.min(255) as u8;
      }
    }
  }
}

/// Hysteresis: mark `mag >= high` as strong (255), `mag >= low` AND
/// 8-connected to strong as edges (255); else 0.
fn hysteresis(buf: &mut [u8], mag_raw: &[i32], low: u8, high: u8, width: u32, height: u32) {
  let w = width as usize;
  let h = height as usize;
  let high = high as i32;
  let low = low as i32;

  // Pass 1: mark strong edges (value 2) and weak edges (value 1).
  for i in 0..(w * h) {
    if buf[i] == 0 {
      continue;
    }
    let m = mag_raw[i];
    if m >= high {
      buf[i] = 2;
    } else if m >= low {
      buf[i] = 1;
    } else {
      buf[i] = 0;
    }
  }

  // Pass 2: propagate strong label via 8-connectivity using a simple
  // worklist-free iterative scan. Two-pass forward/backward converges for
  // dense edge maps; rare pathological layouts may require more iterations,
  // but for typical edge content two passes suffice.
  for _ in 0..2 {
    // Forward.
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        let idx = y * w + x;
        if buf[idx] != 1 {
          continue;
        }
        for (dy, dx) in [(-1i32, -1i32), (-1, 0), (-1, 1), (0, -1)] {
          let ny = (y as i32 + dy) as usize;
          let nx = (x as i32 + dx) as usize;
          if buf[ny * w + nx] == 2 {
            buf[idx] = 2;
            break;
          }
        }
      }
    }
    // Backward.
    for y in (1..h - 1).rev() {
      for x in (1..w - 1).rev() {
        let idx = y * w + x;
        if buf[idx] != 1 {
          continue;
        }
        for (dy, dx) in [(1i32, 1i32), (1, 0), (1, -1), (0, 1)] {
          let ny = (y as i32 + dy) as usize;
          let nx = (x as i32 + dx) as usize;
          if buf[ny * w + nx] == 2 {
            buf[idx] = 2;
            break;
          }
        }
      }
    }
  }

  // Finalize: 2 → 255, anything else → 0.
  for v in buf.iter_mut() {
    *v = if *v == 2 { 255 } else { 0 };
  }
}

/// Separable morphological dilation with a `k × k` square kernel.
/// Horizontal pass → `tmp`, vertical pass → `out`.
fn dilate(input: &[u8], out: &mut [u8], tmp: &mut [u8], width: u32, height: u32, kernel: u32) {
  let w = width as usize;
  let h = height as usize;
  let half = (kernel / 2) as usize;

  // Horizontal pass: tmp[y, x] = max over x' in [x-half, x+half] of input[y, x'].
  for y in 0..h {
    let row_in = &input[y * w..y * w + w];
    let row_out = &mut tmp[y * w..y * w + w];
    for x in 0..w {
      let lo = x.saturating_sub(half);
      let hi = (x + half + 1).min(w);
      let mut m = 0u8;
      for xx in lo..hi {
        if row_in[xx] > m {
          m = row_in[xx];
        }
      }
      row_out[x] = m;
    }
  }

  // Vertical pass: out[y, x] = max over y' in [y-half, y+half] of tmp[y', x].
  for y in 0..h {
    let lo = y.saturating_sub(half);
    let hi = (y + half + 1).min(h);
    for x in 0..w {
      let mut m = 0u8;
      for yy in lo..hi {
        let v = tmp[yy * w + x];
        if v > m {
          m = v;
        }
      }
      out[y * w + x] = m;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::num::NonZeroU32;

  const fn nz32(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
      Some(v) => v,
      None => panic!("zero"),
    }
  }

  fn tb() -> Timebase {
    Timebase::new(1, nz32(1000))
  }

  fn luma_frame<'a>(data: &'a [u8], w: u32, h: u32, pts: i64) -> LumaFrame<'a> {
    LumaFrame::new(data, w, h, w, Timestamp::new(pts, tb()))
  }

  #[test]
  fn components_sum_abs() {
    let c = Components::new(1.0, -2.0, 0.5, 0.0);
    assert_eq!(c.sum_abs(), 3.5);
  }

  #[test]
  fn components_builders_round_trip() {
    let c = Components::new(0.0, 0.0, 0.0, 0.0)
      .with_delta_hue(1.0)
      .with_delta_sat(2.0)
      .with_delta_lum(3.0)
      .with_delta_edges(4.0);
    assert_eq!(c.delta_hue(), 1.0);
    assert_eq!(c.delta_sat(), 2.0);
    assert_eq!(c.delta_lum(), 3.0);
    assert_eq!(c.delta_edges(), 4.0);

    let mut c = Components::default();
    c.set_delta_hue(5.0).set_delta_edges(6.0);
    assert_eq!(c.delta_hue(), 5.0);
    assert_eq!(c.delta_edges(), 6.0);
  }

  #[test]
  fn try_new_rejects_zero_weights() {
    let opts = Options::default().with_weights(Components::new(0.0, 0.0, 0.0, 0.0));
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::ZeroWeights);
  }

  #[test]
  fn try_new_rejects_even_kernel() {
    let opts = Options::default().with_kernel_size(Some(4));
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::InvalidKernelSize(4));
  }

  #[test]
  fn bgr_to_hsv_pure_red() {
    // Pure red: R=255, G=0, B=0 → H=0, S=255, V=255.
    let (h, s, v) = bgr_to_hsv_pixel(0.0, 0.0, 255.0);
    assert_eq!(h, 0);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
  }

  #[test]
  fn bgr_to_hsv_pure_green() {
    // Pure green: H=60° (in 0..359) → 30 in OpenCV's 0..179 encoding.
    let (h, s, v) = bgr_to_hsv_pixel(0.0, 255.0, 0.0);
    assert_eq!(h, 60);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
  }

  #[test]
  fn bgr_to_hsv_pure_blue() {
    // Pure blue: H=240° → 120.
    let (h, s, v) = bgr_to_hsv_pixel(255.0, 0.0, 0.0);
    assert_eq!(h, 120);
    assert_eq!(s, 255);
    assert_eq!(v, 255);
  }

  #[test]
  fn bgr_to_hsv_grayscale() {
    // Grayscale: S=0, V=gray.
    let (h, s, v) = bgr_to_hsv_pixel(128.0, 128.0, 128.0);
    assert_eq!(h, 0);
    assert_eq!(s, 0);
    assert_eq!(v, 128);
  }

  #[test]
  fn median_u8_basic() {
    let v = vec![1u8, 2, 3, 4, 5];
    assert_eq!(median_u8(&v), 3);
    let v = vec![10u8; 100];
    assert_eq!(median_u8(&v), 10);
  }

  #[test]
  fn auto_kernel_size_reasonable() {
    assert_eq!(auto_kernel_size(1920, 1080), 13);
    assert_eq!(auto_kernel_size(1280, 720), 9);
    assert_eq!(auto_kernel_size(640, 360), 7);
  }

  #[test]
  fn identical_luma_frames_zero_score() {
    let opts = Options::default()
      .with_weights(LUMA_ONLY_WEIGHTS)
      .with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);
    let buf = vec![128u8; 32 * 32];
    assert!(det.process_luma(luma_frame(&buf, 32, 32, 0)).is_none());
    assert!(det.process_luma(luma_frame(&buf, 32, 32, 33)).is_none());
    assert_eq!(det.last_score(), Some(0.0));
  }

  #[test]
  fn very_different_luma_frames_exceed_threshold() {
    let opts = Options::default()
      .with_weights(LUMA_ONLY_WEIGHTS)
      .with_min_duration(Duration::from_millis(0))
      .with_threshold(10.0); // lower than default so we actually trip it
    let mut det = Detector::new(opts);
    let a = vec![0u8; 32 * 32];
    let b = vec![255u8; 32 * 32];
    det.process_luma(luma_frame(&a, 32, 32, 0));
    let cut = det.process_luma(luma_frame(&b, 32, 32, 33));
    assert!(
      cut.is_some(),
      "black→white at 32×32 should exceed threshold=10"
    );
  }

  #[test]
  fn initial_cut_true_emits_first_detected_cut() {
    let opts = Options::default()
      .with_weights(LUMA_ONLY_WEIGHTS)
      .with_threshold(10.0)
      .with_initial_cut(true);
    // min_duration = 1 s by default; with initial_cut=true the seed
    // is shifted into the virtual past so the first cut can fire at ts=33.
    let mut det = Detector::new(opts);
    let a = vec![0u8; 32 * 32];
    let b = vec![255u8; 32 * 32];
    det.process_luma(luma_frame(&a, 32, 32, 0));
    let cut = det.process_luma(luma_frame(&b, 32, 32, 33));
    assert!(cut.is_some(), "first cut should fire with initial_cut=true");
  }

  #[test]
  fn initial_cut_false_suppresses_first_detected_cut() {
    let opts = Options::default()
      .with_weights(LUMA_ONLY_WEIGHTS)
      .with_threshold(10.0)
      .with_filter_mode(FilterMode::Suppress)
      .with_initial_cut(false);
    let mut det = Detector::new(opts);
    let a = vec![0u8; 32 * 32];
    let b = vec![255u8; 32 * 32];
    det.process_luma(luma_frame(&a, 32, 32, 0));
    // Rapid (33 ms) cut — with initial_cut=false and min_duration=1s,
    // should be suppressed.
    let cut = det.process_luma(luma_frame(&b, 32, 32, 33));
    assert!(
      cut.is_none(),
      "first cut should be suppressed with initial_cut=false"
    );
  }

  #[test]
  fn clear_resets_state() {
    let opts = Options::default()
      .with_weights(LUMA_ONLY_WEIGHTS)
      .with_threshold(10.0)
      .with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);
    let a = vec![0u8; 32 * 32];
    let b = vec![255u8; 32 * 32];
    det.process_luma(luma_frame(&a, 32, 32, 0));
    det.process_luma(luma_frame(&b, 32, 32, 33));
    assert!(det.last_score().is_some());

    det.clear();
    assert!(det.last_score().is_none());
    // First frame after clear: no cut, re-seeds state.
    assert!(
      det
        .process_luma(luma_frame(&a, 32, 32, 1_000_000))
        .is_none()
    );
  }
}
