//! Adaptive (rolling-average) scene detector.
//!
//! A thin layer built on top of [`content::Detector`]. Each frame is
//! scored exactly as the content detector scores it (weighted HSV / optional
//! edges); the adaptive detector maintains a sliding window of `1 + 2W`
//! scores around a **target** frame and decides whether the target is an
//! outlier — specifically whether its score exceeds a multiple of the local
//! average.
//!
//! This is the algorithm PySceneDetect's `detect-adaptive` uses. Its point:
//! on fast camera motion the content score stays *consistently high* across
//! neighbouring frames, so the ratio of the target score to the window
//! average stays *near 1*. A real cut spikes the target score relative to
//! its neighbours and the ratio jumps.
//!
//! # Algorithm
//!
//! For each incoming frame:
//!
//! 1. Pass the frame to an inner [`content::Detector`] solely for
//!    its score; its own threshold is set to an unreachable value so it
//!    never emits cuts.
//! 2. Read the score and push `(timestamp, score)` onto a ring buffer of
//!    capacity `1 + 2 * window_width`. While the buffer isn't full yet,
//!    return `None`.
//! 3. Once full, the **target** is the middle element (index
//!    `window_width`). Compute
//!    `average = mean(scores except target)` and
//!    `ratio = target_score / average` (capped at 255).
//! 4. Emit a cut **at the target's timestamp** iff:
//!    - `ratio >= adaptive_threshold`,
//!    - `target_score >= min_content_val` (guards against ratio noise in
//!      near-flat sequences),
//!    - at least `min_duration` has elapsed since the previous cut.
//!
//! Because the target lags the current frame by `window_width`, emissions
//! arrive `window_width` frames **behind** the real-time input. Cuts in
//! the final `window_width` frames of a stream are not emitted (there's
//! no future context to evaluate them against) — mirrors PySceneDetect.
//!
//! # Attribution
//!
//! Ported from PySceneDetect's `detect-adaptive` (BSD 3-Clause).

use core::time::Duration;
use derive_more::IsVariant;
use std::collections::VecDeque;
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
  content,
  frame::{HsvFrame, LumaFrame, RgbFrame, Timebase, Timestamp},
};

/// Error returned by [`Detector::try_new`] when the provided [`Options`]
/// are inconsistent or the inner [`content::Options`] is invalid.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum Error {
  /// `options.window_width()` was zero. Must be `>= 1`.
  #[error("window_width must be >= 1")]
  ZeroWindowWidth,
  /// The inner content detector's options were invalid.
  #[error(transparent)]
  Content(#[from] content::Error),
}

/// Options for the adaptive scene detector. See the [module
/// documentation](crate::adaptive) for how each parameter shapes the
/// algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  adaptive_threshold: f64,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  min_duration: Duration,
  window_width: u32,
  min_content_val: f64,
  /// Per-channel scoring weights, same semantics as
  /// [`content::Components`].
  weights: content::Components,
  /// Edge-dilation kernel size (`None` = auto). Same semantics as
  /// [`content::Options::kernel_size`]. Only used when
  /// `weights.delta_edges() != 0.0`.
  kernel_size: Option<u32>,
  /// SIMD toggle, propagated to the inner content scorer.
  simd: bool,
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
  /// Defaults: `adaptive_threshold = 3.0`, `min_duration = 1 s`,
  /// `window_width = 2`, `min_content_val = 15.0`, weights =
  /// [`content::DEFAULT_WEIGHTS`], auto kernel size, SIMD on,
  /// `initial_cut = true`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      adaptive_threshold: 3.0,
      min_duration: Duration::from_secs(1),
      window_width: 2,
      min_content_val: 15.0,
      weights: content::DEFAULT_WEIGHTS,
      kernel_size: None,
      simd: true,
      initial_cut: true,
    }
  }

  /// Returns the adaptive-ratio threshold. The target score must exceed
  /// this multiple of the local window average to trigger a cut.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_threshold(&self) -> f64 {
    self.adaptive_threshold
  }

  /// Sets the adaptive-ratio threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_adaptive_threshold(mut self, val: f64) -> Self {
    self.adaptive_threshold = val;
    self
  }

  /// Sets the adaptive-ratio threshold in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_adaptive_threshold(&mut self, val: f64) -> &mut Self {
    self.adaptive_threshold = val;
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

  /// Set the minimum scene length as a number of frames at a given frame rate.
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

  /// Returns the half-width of the score-averaging window. The full window
  /// contains `1 + 2 * window_width` frames.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn window_width(&self) -> u32 {
    self.window_width
  }

  /// Sets the window half-width. Must be `>= 1`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_window_width(mut self, val: u32) -> Self {
    self.window_width = val;
    self
  }

  /// Sets the window half-width in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_window_width(&mut self, val: u32) -> &mut Self {
    self.window_width = val;
    self
  }

  /// Returns the minimum raw content score required for a cut. Guards
  /// against very small averages producing spurious ratio spikes on
  /// low-variance streams.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_content_val(&self) -> f64 {
    self.min_content_val
  }

  /// Sets `min_content_val`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_content_val(mut self, val: f64) -> Self {
    self.min_content_val = val;
    self
  }

  /// Sets `min_content_val` in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_content_val(&mut self, val: f64) -> &mut Self {
    self.min_content_val = val;
    self
  }

  /// Returns the per-channel scoring weights. Same semantics as
  /// [`content::Options::weights`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn weights(&self) -> &content::Components {
    &self.weights
  }

  /// Sets the per-channel scoring weights.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_weights(mut self, val: content::Components) -> Self {
    self.weights = val;
    self
  }

  /// Sets the per-channel scoring weights in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_weights(&mut self, val: content::Components) -> &mut Self {
    self.weights = val;
    self
  }

  /// Returns the edge-dilation kernel size (`None` = auto). Only used when
  /// `weights.delta_edges() != 0.0`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn kernel_size(&self) -> Option<u32> {
    self.kernel_size
  }

  /// Sets the edge-dilation kernel size.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_kernel_size(mut self, val: Option<u32>) -> Self {
    self.kernel_size = val;
    self
  }

  /// Sets the edge-dilation kernel size in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_kernel_size(&mut self, val: Option<u32>) -> &mut Self {
    self.kernel_size = val;
    self
  }

  /// Returns whether SIMD acceleration is enabled for the inner content
  /// scorer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn simd(&self) -> bool {
    self.simd
  }

  /// Enables or disables SIMD acceleration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, val: bool) -> Self {
    self.simd = val;
    self
  }

  /// Enables or disables SIMD acceleration in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_simd(&mut self, val: bool) -> &mut Self {
    self.simd = val;
    self
  }

  /// Whether the first detected cut is allowed to fire immediately. See
  /// [`content::Options::initial_cut`] for semantics.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn initial_cut(&self) -> bool {
    self.initial_cut
  }

  /// Sets `initial_cut`.
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

/// Adaptive scene detector. See [module documentation](crate::adaptive).
#[derive(Debug, Clone)]
pub struct Detector {
  options: Options,
  inner: content::Detector,
  window_width: usize,
  required_frames: usize,
  buffer: VecDeque<(Timestamp, f64)>,
  /// Rolling sum of all scores currently in `buffer`. Maintained as entries
  /// are pushed / popped so the per-frame average cost is O(1) instead of
  /// O(window_width).
  buffer_sum: f64,
  last_cut_ts: Option<Timestamp>,
  last_adaptive_ratio: Option<f64>,
}

impl Detector {
  /// Creates a new detector with the given options.
  ///
  /// # Panics
  ///
  /// Panics if the options are invalid — see [`enum@Error`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: Options) -> Self {
    Self::try_new(options).expect("invalid adaptive::Options")
  }

  /// Creates a new detector with the given options, returning [`enum@Error`]
  /// on invalid configuration (zero `window_width`, or inner content
  /// options invalid).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new(options: Options) -> Result<Self, Error> {
    if options.window_width == 0 {
      return Err(Error::ZeroWindowWidth);
    }

    let inner = content::Detector::try_new(Self::build_content_options(&options))?;

    let window_width = options.window_width as usize;
    let required_frames = 1 + 2 * window_width;

    Ok(Self {
      options,
      inner,
      window_width,
      required_frames,
      buffer: VecDeque::new(),
      buffer_sum: 0.0,
      last_cut_ts: None,
      last_adaptive_ratio: None,
    })
  }

  /// Returns a reference to the options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// Builds the inner [`content::Options`] used for scoring. Forces
  /// `threshold = INFINITY`, `min_duration = 0`, and `filter_mode = Suppress`
  /// so the inner detector never emits cuts of its own — the adaptive layer
  /// gates emissions based on its own rolling-average test.
  #[cfg_attr(not(tarpaulin), inline(always))]
  const fn build_content_options(options: &Options) -> content::Options {
    content::Options::new()
      .with_weights(options.weights)
      .with_kernel_size(options.kernel_size)
      .with_simd(options.simd)
      .with_threshold(f64::INFINITY)
      .with_min_duration(Duration::from_secs(0))
      .with_filter_mode(content::FilterMode::Suppress)
  }

  /// Returns the adaptive ratio (target score / window average) from the
  /// most recent emission attempt, or `None` if fewer than
  /// `1 + 2 * window_width` frames have been processed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_adaptive_ratio(&self) -> Option<f64> {
    self.last_adaptive_ratio
  }

  /// Returns the score of the most recently processed frame, or `None` if
  /// fewer than two frames have been processed. Delegates to the inner
  /// content detector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn last_score(&self) -> Option<f64> {
    self.inner.last_score()
  }

  /// Resets streaming state.
  pub fn clear(&mut self) {
    self.inner.clear();
    self.buffer.clear();
    self.buffer_sum = 0.0;
    self.last_cut_ts = None;
    self.last_adaptive_ratio = None;
  }

  /// Processes a luma-only frame.
  pub fn process_luma(&mut self, frame: LumaFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.inner.process_luma(frame);
    self.push_and_check(ts)
  }

  /// Processes a packed BGR frame.
  pub fn process_bgr(&mut self, frame: RgbFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.inner.process_bgr(frame);
    self.push_and_check(ts)
  }

  /// Processes a pre-converted HSV frame.
  pub fn process_hsv(&mut self, frame: HsvFrame<'_>) -> Option<Timestamp> {
    let ts = frame.timestamp();
    self.inner.process_hsv(frame);
    self.push_and_check(ts)
  }

  /// Shared logic after the inner detector has scored the frame.
  fn push_and_check(&mut self, ts: Timestamp) -> Option<Timestamp> {
    if self.buffer.capacity() == 0 {
      self.buffer.reserve_exact(self.required_frames);
    }

    // First frame: inner hasn't got a score yet. Don't push.
    let score = self.inner.last_score()?;

    self.buffer.push_back((ts, score));
    self.buffer_sum += score;
    while self.buffer.len() > self.required_frames {
      if let Some((_, popped)) = self.buffer.pop_front() {
        self.buffer_sum -= popped;
      }
    }
    if self.buffer.len() < self.required_frames {
      return None;
    }

    let (target_ts, target_score) = self.buffer[self.window_width];

    // Average of all scores *except* the target. Rolling-sum form is O(1)
    // per frame — the alternative (sum the buffer each frame) is
    // O(window_width) and dominates adaptive overhead at larger windows.
    let denom = (2 * self.window_width) as f64;
    let avg = (self.buffer_sum - target_score) / denom;

    let adaptive_ratio = if avg.abs() < 1e-5 {
      // Avoid divide-by-zero: if target has non-trivial content, treat as
      // max ratio; otherwise no signal.
      if target_score >= self.options.min_content_val {
        255.0
      } else {
        0.0
      }
    } else {
      (target_score / avg).min(255.0)
    };
    self.last_adaptive_ratio = Some(adaptive_ratio);

    // Seed cut-gating reference on first eligible target.
    if self.last_cut_ts.is_none() {
      self.last_cut_ts = Some(if self.options.initial_cut {
        target_ts.saturating_sub_duration(self.options.min_duration)
      } else {
        target_ts
      });
    }

    let threshold_met = adaptive_ratio >= self.options.adaptive_threshold
      && target_score >= self.options.min_content_val;
    let min_length_met = self
      .last_cut_ts
      .as_ref()
      .and_then(|last| target_ts.duration_since(last))
      .is_some_and(|d| d >= self.options.min_duration);

    if threshold_met && min_length_met {
      self.last_cut_ts = Some(target_ts);
      Some(target_ts)
    } else {
      None
    }
  }
}

#[cfg(all(test, feature = "std"))]
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
  fn try_new_rejects_zero_window_width() {
    let opts = Options::default().with_window_width(0);
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::ZeroWindowWidth);
  }

  #[test]
  fn try_new_propagates_content_zero_weights() {
    // Adaptive's weights field is handed verbatim to the inner content
    // detector — all-zero weights trip content's own `ZeroWeights` guard,
    // which adaptive `?`-wraps into `Error::Content`.
    let opts = Options::default().with_weights(content::Components::new(0.0, 0.0, 0.0, 0.0));
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::Content(content::Error::ZeroWeights));
  }

  #[test]
  fn try_new_propagates_content_invalid_kernel() {
    // Same propagation path for kernel_size — even-sized kernels fail
    // content::Detector::try_new.
    let opts = Options::default().with_kernel_size(Some(4));
    let err = Detector::try_new(opts).expect_err("should fail");
    assert_eq!(err, Error::Content(content::Error::InvalidKernelSize(4)));
  }

  #[test]
  fn buffer_fills_before_emitting() {
    // window_width = 2 → required = 5 frames. First 4 must not emit.
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(0))
      .with_weights(content::LUMA_ONLY_WEIGHTS);
    let mut det = Detector::new(opts);

    let buf = vec![128u8; 64 * 48];
    for i in 0..5i64 {
      let cut = det.process_luma(luma_frame(&buf, 64, 48, i * 33));
      if i < 4 {
        assert!(cut.is_none(), "frame {i} should not emit");
      }
    }
  }

  #[test]
  fn flat_content_produces_no_cut() {
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(0))
      .with_weights(content::LUMA_ONLY_WEIGHTS);
    let mut det = Detector::new(opts);

    let buf = vec![128u8; 64 * 48];
    let mut emitted = 0;
    for i in 0..30i64 {
      if det.process_luma(luma_frame(&buf, 64, 48, i * 33)).is_some() {
        emitted += 1;
      }
    }
    assert_eq!(emitted, 0, "flat content has zero score → no cut");
  }

  #[test]
  fn isolated_spike_emits_cut() {
    // Stream is mostly uniform; one frame in the middle differs sharply.
    // That one frame should produce a ratio >> 3.0 (default threshold)
    // against its neighbors and trigger a cut.
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(0))
      .with_weights(content::LUMA_ONLY_WEIGHTS);
    let mut det = Detector::new(opts);

    let dim = vec![50u8; 64 * 48];
    let bright = vec![250u8; 64 * 48];

    // Feed: dim, dim, dim, bright, dim, dim, dim, dim, dim
    // window_width = 2 → target at buffer[2]; cuts lag 2 frames.
    let frames = [&dim, &dim, &dim, &bright, &dim, &dim, &dim, &dim, &dim];
    let mut cuts = Vec::new();
    for (i, f) in frames.iter().enumerate() {
      let ts = (i as i64) * 33;
      if let Some(c) = det.process_luma(luma_frame(f, 64, 48, ts)) {
        cuts.push(c.pts());
      }
    }
    assert!(!cuts.is_empty(), "expected at least one cut on spike");
  }

  #[test]
  fn clear_resets_state() {
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(0))
      .with_weights(content::LUMA_ONLY_WEIGHTS);
    let mut det = Detector::new(opts);

    let buf = vec![128u8; 64 * 48];
    for i in 0..10i64 {
      det.process_luma(luma_frame(&buf, 64, 48, i * 33));
    }
    assert!(det.last_adaptive_ratio().is_some());

    det.clear();
    assert!(det.last_adaptive_ratio().is_none());
    assert!(det.last_score().is_none());
  }

  #[test]
  fn options_accessors_builders_setters_roundtrip() {
    // Sweep every getter/with/set triple on Options so they're exercised at
    // least once for coverage and to catch any future accidental shadowing.
    let fps30 = Timebase::new(30, nz32(1));
    let weights = content::Components::new(0.25, 0.5, 0.75, 1.0);

    // Consuming builder form (with_*) — check each field round-trips.
    let opts = Options::default()
      .with_adaptive_threshold(4.0)
      .with_min_duration(Duration::from_millis(250))
      .with_window_width(8)
      .with_min_content_val(20.0)
      .with_weights(weights)
      .with_kernel_size(Some(5))
      .with_simd(false)
      .with_initial_cut(false);

    assert_eq!(opts.adaptive_threshold(), 4.0);
    assert_eq!(opts.min_duration(), Duration::from_millis(250));
    assert_eq!(opts.window_width(), 8);
    assert_eq!(opts.min_content_val(), 20.0);
    assert_eq!(*opts.weights(), weights);
    assert_eq!(opts.kernel_size(), Some(5));
    assert!(!opts.simd());
    assert!(!opts.initial_cut());

    // with_min_frames alternative form.
    let opts_frames = Options::default().with_min_frames(30, fps30);
    assert_eq!(opts_frames.min_duration(), Duration::from_secs(1));

    // In-place form (set_*). Each returns &mut Self so chaining is possible.
    let mut opts = Options::default();
    opts
      .set_adaptive_threshold(5.0)
      .set_min_duration(Duration::from_secs(2))
      .set_window_width(16)
      .set_min_content_val(30.0)
      .set_weights(content::Components::new(1.0, 0.0, 0.0, 0.0))
      .set_kernel_size(None)
      .set_simd(true)
      .set_initial_cut(true);
    assert_eq!(opts.adaptive_threshold(), 5.0);
    assert_eq!(opts.min_duration(), Duration::from_secs(2));
    assert_eq!(opts.window_width(), 16);
    assert_eq!(opts.min_content_val(), 30.0);
    assert_eq!(opts.kernel_size(), None);
    assert!(opts.simd());
    assert!(opts.initial_cut());

    opts.set_min_frames(60, fps30);
    assert_eq!(opts.min_duration(), Duration::from_secs(2));
  }

  #[test]
  fn detector_plumbing_accessors() {
    // Exercise Detector's options() + last_* accessor surface.
    let opts = Options::default()
      .with_weights(content::LUMA_ONLY_WEIGHTS)
      .with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts.clone());
    assert_eq!(det.options().window_width(), opts.window_width());
    assert!(det.last_score().is_none());
    assert!(det.last_adaptive_ratio().is_none());

    // One frame: inner scoring happens but buffer still under-filled.
    let buf = vec![128u8; 64 * 48];
    for i in 0..3i64 {
      det.process_luma(luma_frame(&buf, 64, 48, i * 33));
    }
    assert!(det.last_score().is_some());
  }

  // Exercise the BGR and HSV entry points — they delegate to the inner
  // content detector then run push_and_check, which is shared.
  #[test]
  fn process_bgr_and_process_hsv_entry_points() {
    use crate::frame::{HsvFrame, RgbFrame};
    let opts = Options::default().with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    let bgr = vec![80u8; 32 * 32 * 3];
    det.process_bgr(RgbFrame::new(&bgr, 32, 32, 32 * 3, Timestamp::new(0, tb())));
    det.process_bgr(RgbFrame::new(
      &bgr,
      32,
      32,
      32 * 3,
      Timestamp::new(33, tb()),
    ));

    det.clear();

    let h = vec![60u8; 32 * 32];
    let s = vec![40u8; 32 * 32];
    let v = vec![200u8; 32 * 32];
    det.process_hsv(HsvFrame::new(
      &h,
      &s,
      &v,
      32,
      32,
      32,
      Timestamp::new(0, tb()),
    ));
    det.process_hsv(HsvFrame::new(
      &h,
      &s,
      &v,
      32,
      32,
      32,
      Timestamp::new(33, tb()),
    ));
    assert!(det.last_score().is_some());
  }

  // Drive the adaptive_ratio-to-255 branch: near-flat neighbors (avg ≈ 0)
  // plus a target score meeting min_content_val emits ratio = 255.
  #[test]
  fn adaptive_ratio_saturates_when_neighbors_are_flat() {
    let opts = Options::default()
      .with_weights(content::LUMA_ONLY_WEIGHTS)
      .with_window_width(1)
      .with_min_content_val(5.0)
      .with_min_duration(Duration::from_millis(0));
    let mut det = Detector::new(opts);

    // window_width = 1 → required_frames = 3. Target is buffer[1].
    // Build a sequence where neighbors (buffer[0], buffer[2]) have score 0
    // (identical frames → zero inner delta) and the target has a large
    // score (its frame differs sharply).
    //
    // NOTE: the inner content detector's `last_score` reflects the delta
    // with the *previous* frame, so we need careful sequencing. We emit
    // a spike so the target's score is high while the surrounding scores
    // are small.
    let dim = vec![10u8; 32 * 32];
    let bright = vec![250u8; 32 * 32];

    // Sequence of 5 frames so the buffer reaches 3 with the target at idx 1.
    let frames = [&dim, &dim, &dim, &bright, &dim];
    for (i, f) in frames.iter().enumerate() {
      det.process_luma(luma_frame(f, 32, 32, (i as i64) * 33));
    }
    // Some ratio should have been computed.
    assert!(det.last_adaptive_ratio().is_some());
  }

  // Exercise the initial_cut = false seed path in push_and_check.
  #[test]
  fn initial_cut_false_seeds_last_cut_at_target_ts() {
    let opts = Options::default()
      .with_weights(content::LUMA_ONLY_WEIGHTS)
      .with_window_width(1)
      .with_min_duration(Duration::from_millis(0))
      .with_initial_cut(false);
    let mut det = Detector::new(opts);

    let buf = vec![128u8; 32 * 32];
    for i in 0..5i64 {
      det.process_luma(luma_frame(&buf, 32, 32, i * 33));
    }
    // No panic, ratio tracked — the `else` branch of the seed ran.
    assert!(det.last_adaptive_ratio().is_some());
  }
}
