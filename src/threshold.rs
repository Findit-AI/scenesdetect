//! Intensity-threshold scene detection — fade-in / fade-out transitions.
//!
//! This module implements [`Detector`](crate::threshold::Detector), a port
//! of PySceneDetect's `detect-threshold` algorithm. Unlike the
//! frame-difference detectors ([`histogram`](crate::histogram),
//! [`phash`](crate::phash)), this one looks at the **absolute mean
//! brightness** of each frame and fires when the mean crosses a threshold
//! in one direction and then the other.
//!
//! Typical use: detecting fades-to-black between scenes in films.
//!
//! # Algorithm
//!
//! The detector runs a two-state machine, with the state determined by the
//! current frame's mean intensity relative to `threshold`:
//!
//! - **`In`** — we're inside a lit scene (mean ≥ threshold, for `Floor`).
//! - **`Out`** — we're in a fade-to-black (mean < threshold, for `Floor`).
//!
//! For each frame:
//!
//! 1. **Compute mean intensity.** For [`LumaFrame`](crate::frame::LumaFrame)
//!    inputs, the mean of the Y plane. For
//!    [`RgbFrame`](crate::frame::RgbFrame) inputs, the mean of all
//!    3 × W × H bytes — mirroring Python's `numpy.mean(frame_img)` over a
//!    BGR image.
//! 2. **Check for a state transition.**
//!    - `In → Out`: store this frame's timestamp as the fade-out start.
//!    - `Out → In`: we just completed a full fade cycle. Emit a cut
//!      **interpolated between the fade-out and fade-in endpoints** by
//!      [`Options::fade_bias`](crate::threshold::Options::fade_bias), gated
//!      by [`Options::min_duration`](crate::threshold::Options::min_duration).
//!
//! The interpolation is:
//!
//! ```text
//! cut_time = f_out + (f_in - f_out) * (1 + fade_bias) / 2
//! ```
//!
//! so `fade_bias = -1` places the cut at the fade-out frame, `0` at the
//! midpoint (default), and `+1` at the fade-in frame.
//!
//! # End-of-stream handling
//!
//! If the stream ends while the detector is in `Out` state (fade-to-black
//! without a recovery) and
//! [`Options::add_final_scene`](crate::threshold::Options::add_final_scene)
//! is set, calling
//! [`Detector::finish`](crate::threshold::Detector::finish) emits one final
//! cut at the fade-out frame. This represents "the last scene ended when
//! the video faded out."
//!
//! [`Detector::clear`](crate::threshold::Detector::clear) resets stream
//! state so the same detector instance can be reused for the next video.
//!
//! # [`Method`](crate::threshold::Method) variants
//!
//! - [`Method::Floor`](crate::threshold::Method::Floor) — "dark = below
//!   threshold" (fade to black, default).
//! - [`Method::Ceiling`](crate::threshold::Method::Ceiling) — "bright =
//!   above threshold" (fade to white).
//!
//! # Attribution
//!
//! Ported from PySceneDetect's `detect-threshold` (BSD 3-Clause).
//! See <https://scenedetect.com> for the original implementation.

use core::time::Duration;

use crate::frame::{LumaFrame, RgbFrame, TimeRange, Timebase, Timestamp};

use derive_more::{Display, IsVariant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Which direction of threshold crossing counts as a fade.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum Method {
  /// Fade detected when mean pixel intensity **falls below** `threshold`.
  /// Matches the classic "fade to black" case and is the default.
  #[default]
  Floor,
  /// Fade detected when mean pixel intensity **rises above** `threshold`
  /// (fade to white, or overexposure detection).
  Ceiling,
}

impl Method {
  /// Returns a human-friendly name for this method variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Method::Floor => "floor",
      Method::Ceiling => "ceiling",
    }
  }
}

/// Options for the intensity-threshold scene detector. See the
/// [module docs](crate::threshold) for how each parameter shapes the algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  threshold: u8,
  method: Method,
  fade_bias: f64,
  add_final_scene: bool,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  min_duration: Duration,
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
  /// Defaults: `threshold = 12`, `method = Floor`, `fade_bias = 0.0`,
  /// `add_final_scene = false`, `min_duration = 1 s`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold: 12,
      method: Method::Floor,
      fade_bias: 0.0,
      add_final_scene: false,
      min_duration: Duration::from_secs(1),
      initial_cut: true,
    }
  }

  /// Returns the mean-intensity threshold used for fade detection.
  ///
  /// Interpreted as an 8-bit brightness value in `[0, 255]`. Frames with a
  /// mean below this (for [`Method::Floor`]) are considered "dark".
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold(&self) -> u8 {
    self.threshold
  }

  /// Set the threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold(mut self, val: u8) -> Self {
    self.set_threshold(val);
    self
  }

  /// Set the threshold in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threshold(&mut self, val: u8) -> &mut Self {
    self.threshold = val;
    self
  }

  /// Returns the fade-detection [`Method`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn method(&self) -> Method {
    self.method
  }

  /// Set the method.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_method(mut self, val: Method) -> Self {
    self.set_method(val);
    self
  }

  /// Set the method in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_method(&mut self, val: Method) -> &mut Self {
    self.method = val;
    self
  }

  /// Returns the fade bias, clamped to `[-1.0, 1.0]` at use time.
  ///
  /// Controls cut placement between the fade-out and fade-in frames:
  /// `-1` = at fade-out, `0` = midpoint (default), `+1` = at fade-in.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn fade_bias(&self) -> f64 {
    self.fade_bias
  }

  /// Set the fade bias.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_fade_bias(mut self, val: f64) -> Self {
    self.set_fade_bias(val);
    self
  }

  /// Set the fade bias in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_fade_bias(&mut self, val: f64) -> &mut Self {
    self.fade_bias = val;
    self
  }

  /// Returns whether [`Detector::finish`] will emit a final cut when the
  /// stream ends in the `Out` state.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn add_final_scene(&self) -> bool {
    self.add_final_scene
  }

  /// Set whether to emit a final cut at end-of-stream when in `Out` state.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_add_final_scene(mut self, val: bool) -> Self {
    self.set_add_final_scene(val);
    self
  }

  /// Set whether to emit a final cut at end-of-stream in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_add_final_scene(&mut self, val: bool) -> &mut Self {
    self.add_final_scene = val;
    self
  }

  /// Returns the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_duration(&self) -> Duration {
    self.min_duration
  }

  /// Set the minimum scene duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_duration(mut self, val: Duration) -> Self {
    self.set_min_duration(val);
    self
  }

  /// Set the minimum scene duration in place.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_duration(&mut self, val: Duration) -> &mut Self {
    self.min_duration = val;
    self
  }

  /// Set the minimum scene length as a number of frames at a given frame rate.
  ///
  /// See [`crate::histogram::Options::with_min_frames`] for the semantics.
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
  /// - `true` (default): the first complete fade cycle emits a cut as soon
  ///   as the min-duration gate is satisfied relative to stream start.
  /// - `false`: suppresses cuts until the stream has actually run for at
  ///   least [`Self::min_duration`]. Matches PySceneDetect's default.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn initial_cut(&self) -> bool {
    self.initial_cut
  }

  /// Sets whether the first detected cut may fire immediately.
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

/// Internal state: which side of the threshold the detector is currently on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FadeType {
  /// Mean intensity above threshold (or below, for `Method::Ceiling`).
  In,
  /// Mean intensity below threshold (or above, for `Method::Ceiling`).
  Out,
}

/// Intensity-threshold scene detector. See the
/// [module documentation](crate::threshold) for the algorithm.
#[derive(Debug, Clone)]
pub struct Detector {
  options: Options,
  processed_frame: bool,
  last_scene_cut: Option<Timestamp>,
  /// Timestamp of the frame where the last fade transition occurred.
  last_fade_frame: Option<Timestamp>,
  last_fade_type: FadeType,
  last_avg: Option<f64>,
  /// Fade-out / fade-in endpoints of the most recent emission. Preserved
  /// across [`Self::finish`] so callers can read it after an end-of-stream
  /// cut; only [`Self::clear`] zeroes it.
  last_fade_range: Option<TimeRange>,
}

impl Detector {
  /// Creates a new detector with the given options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(options: Options) -> Self {
    Self {
      options,
      processed_frame: false,
      last_scene_cut: None,
      last_fade_frame: None,
      last_fade_type: FadeType::In,
      last_avg: None,
      last_fade_range: None,
    }
  }

  /// Returns a reference to the options used by this detector.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// Returns the mean intensity of the most recently processed frame, or
  /// `None` if no frame has been processed yet. Useful for diagnostics and
  /// threshold tuning.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_avg(&self) -> Option<f64> {
    self.last_avg
  }

  /// Returns the fade-out / fade-in endpoints of the most recently emitted
  /// cut, or `None` if no cut has fired since the last [`Self::clear`].
  ///
  /// The [`TimeRange`]'s `start` is the fade-out frame's timestamp; `end`
  /// is the fade-in frame's timestamp (both in the fade-out frame's
  /// timebase — `end` is rescaled if timebases differ between frames).
  /// For cuts emitted by [`Self::finish`] there is no matching fade-in, so
  /// the range is degenerate (`start == end == fade_out_ts`).
  ///
  /// `process_*` and `finish` return the single bias-interpolated point
  /// between these two endpoints (see [`Options::fade_bias`]); this
  /// accessor exposes the full range so callers that want the fade
  /// duration — or want to pick a different interpolation — can get both
  /// timestamps without recomputing.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn last_fade_range(&self) -> Option<TimeRange> {
    self.last_fade_range
  }

  /// Processes a luma (Y-plane) frame.
  ///
  /// The per-pixel "intensity" is the 8-bit Y value. Thresholds should be
  /// interpreted in this luma scale.
  pub fn process_luma(&mut self, frame: LumaFrame<'_>) -> Option<Timestamp> {
    let mean = luma_mean(&frame);
    self.process_with_mean(mean, frame.timestamp())
  }

  /// Processes a packed 24-bit RGB (or BGR) frame.
  ///
  /// The per-pixel "intensity" is the average of the three channel bytes —
  /// matching Python's `numpy.mean(frame_img)` over a BGR frame. Because
  /// averaging is channel-order-agnostic, RGB and BGR inputs produce
  /// identical results.
  pub fn process_rgb(&mut self, frame: RgbFrame<'_>) -> Option<Timestamp> {
    let mean = rgb_mean(&frame);
    self.process_with_mean(mean, frame.timestamp())
  }

  /// Signals that the stream has ended at `last_ts`. Returns a final cut if
  /// the stream ended during a fade-out (state = `Out`) and
  /// [`Options::add_final_scene`] is enabled.
  ///
  /// The returned cut is placed at the fade-out frame's timestamp (no bias
  /// applied — there's no matching fade-in to interpolate against).
  ///
  /// `finish` **always calls [`Self::clear`] before returning**, so the same
  /// detector instance is immediately ready for the next video. Subsequent
  /// calls to `finish` without any intervening `process_*` will return
  /// `None` (nothing to finish).
  pub fn finish(&mut self, _last_ts: Timestamp) -> Option<Timestamp> {
    let cut = self.final_cut();
    // If we're emitting a final cut, record a degenerate range at the
    // fade-out frame (no matching fade-in at end-of-stream). This lets
    // callers query `last_fade_range()` after `finish` for consistency
    // with mid-stream emissions.
    let range_after = cut.map(TimeRange::instant);
    self.clear();
    self.last_fade_range = range_after;
    cut
  }

  /// Computes the end-of-stream cut (if any) without mutating state —
  /// [`Self::finish`] calls this, then clears.
  fn final_cut(&self) -> Option<Timestamp> {
    if !self.options.add_final_scene {
      return None;
    }
    if self.last_fade_type != FadeType::Out {
      return None;
    }
    let fade_frame = self.last_fade_frame?;
    // Gate on the cut we're about to emit (`fade_frame`), not on the last
    // observed frame — otherwise a long tail of above-threshold frames
    // after the fade-out would let us emit `fade_frame` even though it's
    // closer than `min_duration` to the previous cut.
    let min_elapsed = match &self.last_scene_cut {
      Some(last) => fade_frame
        .duration_since(last)
        .is_some_and(|d| d >= self.options.min_duration),
      None => true,
    };
    if min_elapsed { Some(fade_frame) } else { None }
  }

  /// Resets the detector's streaming state so it can be reused for the
  /// next video without reallocating.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear(&mut self) {
    self.processed_frame = false;
    self.last_scene_cut = None;
    self.last_fade_frame = None;
    self.last_fade_type = FadeType::In;
    self.last_avg = None;
    self.last_fade_range = None;
  }

  /// Shared state-machine logic, parameterized by the per-frame mean.
  fn process_with_mean(&mut self, mean: f64, ts: Timestamp) -> Option<Timestamp> {
    self.last_avg = Some(mean);
    if self.last_scene_cut.is_none() {
      self.last_scene_cut = Some(if self.options.initial_cut {
        ts.saturating_sub_duration(self.options.min_duration)
      } else {
        ts
      });
    }

    let thresh = self.options.threshold as f64;
    // `dark` means "on the trigger side of the threshold":
    //   Floor   → brightness < threshold
    //   Ceiling → brightness ≥ threshold
    let dark = match self.options.method {
      Method::Floor => mean < thresh,
      Method::Ceiling => mean >= thresh,
    };

    let mut cut: Option<Timestamp> = None;

    if self.processed_frame {
      match self.last_fade_type {
        FadeType::In if dark => {
          // Fade-out just started.
          self.last_fade_type = FadeType::Out;
          self.last_fade_frame = Some(ts);
        }
        FadeType::Out if !dark => {
          // Fade-in completes a fade cycle.
          if let Some(f_out) = self.last_fade_frame {
            let placed = interpolate_cut(f_out, ts, self.options.fade_bias);
            // min_duration is measured from the previously emitted cut to
            // the one we're about to emit (`placed`), so the gate is
            // consistent with what the caller observes.
            let min_elapsed = match &self.last_scene_cut {
              Some(last) => placed
                .duration_since(last)
                .is_some_and(|d| d >= self.options.min_duration),
              None => true,
            };
            if min_elapsed {
              cut = Some(placed);
              self.last_scene_cut = Some(placed);
              // Expose the full [fade_out, fade_in] range for callers who
              // want richer info than the interpolated point. Rescale f_in
              // into f_out's timebase so endpoints share a timebase
              // (rescale_to is a no-op when timebases already match).
              let f_in_same = ts.rescale_to(f_out.timebase());
              self.last_fade_range = Some(TimeRange::new(
                f_out.pts(),
                f_in_same.pts(),
                f_out.timebase(),
              ));
            }
          }
          self.last_fade_type = FadeType::In;
          self.last_fade_frame = Some(ts);
        }
        _ => {}
      }
    } else {
      // First frame: seed the state and the fade reference.
      self.last_fade_frame = Some(ts);
      self.last_fade_type = if dark { FadeType::Out } else { FadeType::In };
      self.processed_frame = true;
    }

    cut
  }
}

/// Mean of the Y plane (same pattern as the histogram detector's inner loop
/// but summing into `u64` — 4K (8.3 M u8 pixels) stays well inside `u64`).
fn luma_mean(frame: &LumaFrame<'_>) -> f64 {
  let data = frame.data();
  let w = frame.width() as usize;
  let h = frame.height() as usize;
  let s = frame.stride() as usize;
  let mut sum: u64 = 0;
  for y in 0..h {
    let row_start = y * s;
    let row = &data[row_start..row_start + w];
    for &v in row {
      sum += v as u64;
    }
  }
  let n = w * h;
  if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

/// Mean of all `width * height * 3` bytes in a packed RGB frame — matches
/// `numpy.mean(frame_img)` over a BGR image in the original Python.
fn rgb_mean(frame: &RgbFrame<'_>) -> f64 {
  let data = frame.data();
  let w = frame.width() as usize;
  let h = frame.height() as usize;
  let s = frame.stride() as usize;
  let row_bytes = w * 3;
  let mut sum: u64 = 0;
  for y in 0..h {
    let row_start = y * s;
    let row = &data[row_start..row_start + row_bytes];
    for &v in row {
      sum += v as u64;
    }
  }
  let n = row_bytes * h;
  if n == 0 { 0.0 } else { sum as f64 / n as f64 }
}

/// Interpolates a cut between the fade-out and fade-in timestamps by the
/// given `bias ∈ [-1, 1]`: `-1` places the cut at `f_out`, `0` at the
/// midpoint, `+1` at `f_in`.
///
/// If the two timestamps have different timebases, `f_in` is rescaled into
/// `f_out`'s timebase first (via [`Timestamp::rescale_to`]). Arithmetic is
/// done in integer PTS units and rounded toward zero.
fn interpolate_cut(f_out: Timestamp, f_in: Timestamp, bias: f64) -> Timestamp {
  let bias = bias.clamp(-1.0, 1.0);
  let f_in_same = if f_in.timebase() == f_out.timebase() {
    f_in
  } else {
    f_in.rescale_to(f_out.timebase())
  };
  let delta = f_in_same.pts() - f_out.pts();
  let lerp = (1.0 + bias) * 0.5;
  let offset = (delta as f64 * lerp) as i64;
  Timestamp::new(f_out.pts() + offset, f_out.timebase())
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
    Timebase::new(1, nz32(1000)) // 1 ms units
  }

  fn luma(data: &[u8], w: u32, h: u32, pts: i64) -> LumaFrame<'_> {
    LumaFrame::new(data, w, h, w, Timestamp::new(pts, tb()))
  }

  fn rgb(data: &[u8], w: u32, h: u32, pts: i64) -> RgbFrame<'_> {
    RgbFrame::new(data, w, h, w * 3, Timestamp::new(pts, tb()))
  }

  #[test]
  fn luma_mean_uniform() {
    let buf = [128u8; 64 * 48];
    let m = luma_mean(&luma(&buf, 64, 48, 0));
    assert!((m - 128.0).abs() < 1e-9);
  }

  #[test]
  fn rgb_mean_uniform() {
    let buf = [64u8; 32 * 24 * 3];
    let m = rgb_mean(&rgb(&buf, 32, 24, 0));
    assert!((m - 64.0).abs() < 1e-9);
  }

  #[test]
  fn rgb_mean_mixed_channels() {
    // Every pixel R=30, G=60, B=150 → per-pixel avg = 80 → frame mean = 80.
    let mut buf = vec![0u8; 4 * 4 * 3];
    for i in 0..(4 * 4) {
      buf[i * 3] = 30;
      buf[i * 3 + 1] = 60;
      buf[i * 3 + 2] = 150;
    }
    let m = rgb_mean(&rgb(&buf, 4, 4, 0));
    assert!((m - 80.0).abs() < 1e-9);
  }

  #[test]
  fn interpolate_cut_midpoint_mixed_timebase() {
    // 1.0 s at 1/1000 timebase, 2.0 s at 1/90000 timebase.
    let f_out = Timestamp::new(1000, Timebase::new(1, nz32(1000)));
    let f_in = Timestamp::new(180_000, Timebase::new(1, nz32(90_000)));
    let cut = interpolate_cut(f_out, f_in, 0.0);
    // Midpoint of 1.0 s and 2.0 s = 1.5 s = 1500 ms in f_out's timebase.
    assert_eq!(cut.pts(), 1500);
    assert_eq!(cut.timebase(), f_out.timebase());
  }

  #[test]
  fn interpolate_cut_bias_bounds() {
    let f_out = Timestamp::new(100, Timebase::new(1, nz32(1000)));
    let f_in = Timestamp::new(200, Timebase::new(1, nz32(1000)));
    assert_eq!(interpolate_cut(f_out, f_in, -1.0).pts(), 100);
    assert_eq!(interpolate_cut(f_out, f_in, 1.0).pts(), 200);
    // Out of range should clamp.
    assert_eq!(interpolate_cut(f_out, f_in, -5.0).pts(), 100);
    assert_eq!(interpolate_cut(f_out, f_in, 5.0).pts(), 200);
  }

  /// Helper: build a uniform luma frame of size 8x8 with given intensity.
  fn uniform_luma(intensity: u8, _pts: i64) -> Vec<u8> {
    vec![intensity; 64]
  }

  #[test]
  fn first_frame_emits_no_cut() {
    let mut det = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));
    // Start dark.
    let buf = uniform_luma(5, 0);
    assert!(det.process_luma(luma(&buf, 8, 8, 0)).is_none());
    assert_eq!(det.last_avg(), Some(5.0));
  }

  #[test]
  fn fade_out_then_fade_in_emits_cut_at_midpoint() {
    // Stream: bright → bright → DARK → DARK → BRIGHT (fade cycle).
    // Defaults: threshold=12, fade_bias=0 → cut at midpoint.
    let mut det = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));

    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    // pts in 1/1000 timebase = ms.
    assert!(det.process_luma(luma(&bright, 8, 8, 0)).is_none());
    assert!(det.process_luma(luma(&bright, 8, 8, 100)).is_none());
    // fade out begins at 200 ms.
    assert!(det.process_luma(luma(&dark, 8, 8, 200)).is_none());
    assert!(det.process_luma(luma(&dark, 8, 8, 300)).is_none());
    // fade in completes at 400 ms → cut placed at midpoint of 200..400 = 300.
    let cut = det.process_luma(luma(&bright, 8, 8, 400));
    assert!(cut.is_some(), "expected cut on fade-in");
    assert_eq!(cut.unwrap().pts(), 300);
  }

  #[test]
  fn fade_bias_places_cut_at_fade_out_or_fade_in() {
    // bias = -1 → cut at fade-out frame.
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_fade_bias(-1.0),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200));
    let cut = det.process_luma(luma(&bright, 8, 8, 400)).unwrap();
    assert_eq!(cut.pts(), 200);

    // bias = +1 → cut at fade-in frame.
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_fade_bias(1.0),
    );
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200));
    let cut = det.process_luma(luma(&bright, 8, 8, 400)).unwrap();
    assert_eq!(cut.pts(), 400);
  }

  #[test]
  fn min_duration_suppresses_cuts() {
    // 1 second gate (default). Time values chosen so the first cycle lands
    // beyond the gate from the seeded `last_scene_cut` (pts=0), but the
    // second cycle falls within the gate after the first cut.
    let mut det = Detector::new(Options::default());
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    // First cycle: seed at 0 ms; fade-out at 1000 ms; fade-in at 1500 ms.
    // Gap from seed = 1500 ms ≥ 1000 ms → cut fires.
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 1000));
    let c1 = det.process_luma(luma(&bright, 8, 8, 1500));
    assert!(c1.is_some(), "first cut should fire (gap >= 1s from seed)");

    // Second cycle immediately after: fade-out at 1600 ms, fade-in at 1700 ms.
    // Gap from last cut (ts=1500) = 200 ms < 1 s → suppressed.
    det.process_luma(luma(&dark, 8, 8, 1600));
    let c2 = det.process_luma(luma(&bright, 8, 8, 1700));
    assert!(c2.is_none(), "second cut should be suppressed within 1s");
  }

  #[test]
  fn ceiling_method_fires_on_rising_edge() {
    // With Method::Ceiling and threshold=200, brightness above 200 = "dark" state.
    let mut det = Detector::new(
      Options::default()
        .with_method(Method::Ceiling)
        .with_threshold(200)
        .with_min_duration(Duration::from_millis(0)),
    );
    let dim = uniform_luma(100, 0);
    let bright = uniform_luma(250, 0);

    det.process_luma(luma(&dim, 8, 8, 0));
    // dim → bright: enter Out.
    det.process_luma(luma(&bright, 8, 8, 100));
    // bright → dim: exit Out → In, cut fires.
    let cut = det.process_luma(luma(&dim, 8, 8, 200));
    assert!(cut.is_some());
  }

  #[test]
  fn last_fade_range_exposes_full_endpoints() {
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_fade_bias(0.0),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200)); // fade-out begins
    let cut = det.process_luma(luma(&bright, 8, 8, 400)).expect("cut"); // fade-in completes

    // Interpolated midpoint.
    assert_eq!(cut.pts(), 300);

    // Full range available via accessor.
    let range = det.last_fade_range().expect("range");
    assert_eq!(range.start_pts(), 200);
    assert_eq!(range.end_pts(), 400);
    assert_eq!(range.timebase(), tb());
    // Duration = 200 ms.
    assert_eq!(range.duration(), Some(Duration::from_millis(200)));
    // Interpolate midpoint matches the emitted cut.
    assert_eq!(range.interpolate(0.5).pts(), 300);
  }

  #[test]
  fn last_fade_range_cleared_by_clear() {
    let mut det = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200));
    det.process_luma(luma(&bright, 8, 8, 400));
    assert!(det.last_fade_range().is_some());
    det.clear();
    assert!(det.last_fade_range().is_none());
  }

  #[test]
  fn last_fade_range_survives_finish_as_instant() {
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_add_final_scene(true),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200)); // fade-out at 200; never recovers
    let final_cut = det.finish(Timestamp::new(400, tb())).expect("final cut");
    assert_eq!(final_cut.pts(), 200);
    // finish emits a degenerate range at the fade-out frame.
    let range = det.last_fade_range().expect("range after finish");
    assert!(range.is_instant());
    assert_eq!(range.start_pts(), 200);
    assert_eq!(range.end_pts(), 200);
  }

  #[test]
  fn finish_emits_final_cut_when_ending_in_fade_out() {
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_add_final_scene(true),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&bright, 8, 8, 100));
    // fade out at 200; stream ends without fade-in.
    det.process_luma(luma(&dark, 8, 8, 200));
    det.process_luma(luma(&dark, 8, 8, 300));

    let final_cut = det.finish(Timestamp::new(400, tb()));
    assert!(final_cut.is_some());
    assert_eq!(final_cut.unwrap().pts(), 200);
  }

  #[test]
  fn finish_returns_none_when_add_final_scene_disabled() {
    let mut det = Detector::new(
      Options::default().with_min_duration(Duration::from_millis(0)),
      // add_final_scene is false by default.
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200));
    assert!(det.finish(Timestamp::new(400, tb())).is_none());
  }

  #[test]
  fn finish_clears_state() {
    // Whether or not a final cut is emitted, finish() must leave the detector
    // in a clean state — `last_avg` reset, no leftover fade reference.
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_add_final_scene(true),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 200));
    assert!(det.last_avg().is_some());

    let final_cut = det.finish(Timestamp::new(400, tb()));
    assert!(final_cut.is_some());
    assert!(
      det.last_avg().is_none(),
      "finish should have cleared last_avg"
    );

    // A second finish with no frames in between is a safe no-op.
    assert!(det.finish(Timestamp::new(500, tb())).is_none());

    // Processing a fresh stream works without an explicit clear().
    assert!(det.process_luma(luma(&bright, 8, 8, 1_000_000)).is_none());
    det.process_luma(luma(&dark, 8, 8, 1_000_200));
    let cut = det.process_luma(luma(&bright, 8, 8, 1_000_400));
    assert!(cut.is_some(), "detector should be reusable after finish()");
  }

  #[test]
  fn finish_returns_none_when_ending_in_fade_in() {
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(0))
        .with_add_final_scene(true),
    );
    let bright = uniform_luma(200, 0);
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&bright, 8, 8, 100));
    assert!(det.finish(Timestamp::new(200, tb())).is_none());
  }

  #[test]
  fn clear_resets_stream_state() {
    let mut det = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    // Video 1: prime, then complete a fade cycle.
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 100));
    let cut1 = det.process_luma(luma(&bright, 8, 8, 200));
    assert!(cut1.is_some());

    det.clear();
    assert!(det.last_avg().is_none());

    // Video 2: start with dark; no cut until a fade-in completes.
    assert!(det.process_luma(luma(&dark, 8, 8, 1_000_000)).is_none());
    // One frame later we cross to bright — that's a fade-in but we came
    // *from* Out at the start, not via a detected In → Out transition, so
    // it completes a fade cycle and emits a cut.
    let cut2 = det.process_luma(luma(&bright, 8, 8, 1_000_100));
    assert!(cut2.is_some(), "cut detection resumes after clear");
  }

  #[test]
  fn min_duration_gate_measured_from_emitted_cut_not_fade_in() {
    // Regression: the min-duration gate is anchored on the *emitted* cut
    // (the interpolated placement between fade-out and fade-in), not on the
    // fade-in frame. Otherwise long fades consume part of the gate window.
    //
    // Schedule (min_duration = 200 ms, fade_bias = 0 so placed = midpoint):
    //   bright(0) dark(100)  -> fade-out starts at 100
    //   bright(200)          -> fade-in; cut1 placed = 150  (midpoint)
    //   dark(250)            -> fade-out starts at 250
    //   bright(300)          -> fade-in; cut2 placed = 275
    //
    // Between cut1 (150) and cut2 (275): 125 ms < 200 ms → cut2 must be
    // suppressed. The previous code set `last_scene_cut = 200` (fade-in),
    // so the gate from the fade-in's POV looked like 300 - 200 = 100 ms,
    // which was also < 200 ms and therefore happened to suppress cut2 in
    // this exact schedule. Stretch the second fade so it's >200 ms from
    // fade-in but <200 ms from the emitted cut to surface the bug:
    //   cut1 placed = 150, cut2 placed = 250 (150 ms apart).
    //   fade-in (201→400) sits 200 ms from fade-in-1 (=200), 250 ms from
    //   the previously-wrongly-recorded fade-in.
    // Concretely: bright(0) dark(100) bright(200) (cut1 @150) dark(300)
    // bright(400) -> cut2 placed = 350.
    //   gate-from-emitted: 350 - 150 = 200  ✅ allowed (exactly min_duration)
    //   gate-from-fade-in: 350 - 200 = 150  ❌ would suppress
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(200))
        .with_fade_bias(0.0),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 100));
    let cut1 = det.process_luma(luma(&bright, 8, 8, 200)).expect("cut1");
    assert_eq!(cut1.pts(), 150);

    det.process_luma(luma(&dark, 8, 8, 300));
    let cut2 = det.process_luma(luma(&bright, 8, 8, 400));
    assert!(
      cut2.is_some(),
      "cut2 should fire — 350 - 150 = 200 ms meets the gate",
    );
    assert_eq!(cut2.unwrap().pts(), 350);
  }

  #[test]
  fn final_cut_gated_on_fade_frame_not_last_ts() {
    // Regression: `finish()`'s min-duration gate compares the emitted
    // `fade_frame` against the previous cut, not the `last_ts` argument.
    // Otherwise a long tail of frames before finish() would let a final
    // cut fire even though its timestamp is too close to the previous one.
    //
    // Schedule (min_duration = 200 ms, fade_bias = 0):
    //   bright(0) dark(100) bright(200)   -> cut1 placed = 150
    //   dark(250)                         -> fade-out at 250, no fade-in
    //   finish(10_000)                    -> last_ts far in the future
    //
    // gate-from-fade_frame: 250 - 150 = 100 < 200 → suppress (correct).
    // gate-from-last_ts:    10000 - 150 huge ≥ 200 → would emit (wrong).
    let mut det = Detector::new(
      Options::default()
        .with_min_duration(Duration::from_millis(200))
        .with_fade_bias(0.0)
        .with_add_final_scene(true),
    );
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 100));
    det.process_luma(luma(&bright, 8, 8, 200));
    det.process_luma(luma(&dark, 8, 8, 250));

    let final_cut = det.finish(Timestamp::new(10_000, tb()));
    assert!(
      final_cut.is_none(),
      "final cut must be suppressed — 250 is only 100 ms from the previous cut (150)"
    );
  }

  #[test]
  fn process_rgb_equivalent_to_luma_for_uniform_frames() {
    // Uniform 100 RGB → mean 100; uniform 100 Y → mean 100. Same state
    // transitions, same cut placement.
    let mut det_l = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));
    let mut det_r = Detector::new(Options::default().with_min_duration(Duration::from_millis(0)));

    let luma_bright = uniform_luma(200, 0);
    let luma_dark = uniform_luma(5, 0);
    let rgb_bright = vec![200u8; 64 * 3];
    let rgb_dark = vec![5u8; 64 * 3];

    det_l.process_luma(luma(&luma_bright, 8, 8, 0));
    det_l.process_luma(luma(&luma_dark, 8, 8, 200));
    let cut_l = det_l.process_luma(luma(&luma_bright, 8, 8, 400));

    det_r.process_rgb(rgb(&rgb_bright, 8, 8, 0));
    det_r.process_rgb(rgb(&rgb_dark, 8, 8, 200));
    let cut_r = det_r.process_rgb(rgb(&rgb_bright, 8, 8, 400));

    assert_eq!(cut_l.map(|t| t.pts()), cut_r.map(|t| t.pts()));
  }

  #[test]
  fn method_as_str_all_variants() {
    assert_eq!(Method::Floor.as_str(), "floor");
    assert_eq!(Method::Ceiling.as_str(), "ceiling");
  }

  #[test]
  fn options_accessors_builders_setters_roundtrip() {
    let fps30 = Timebase::new(30, nz32(1));

    // Consuming builder form — each field round-trips.
    let opts = Options::default()
      .with_threshold(50)
      .with_method(Method::Ceiling)
      .with_fade_bias(0.25)
      .with_add_final_scene(true)
      .with_min_duration(Duration::from_millis(750))
      .with_initial_cut(false);
    assert_eq!(opts.threshold(), 50);
    assert_eq!(opts.method(), Method::Ceiling);
    assert_eq!(opts.fade_bias(), 0.25);
    assert!(opts.add_final_scene());
    assert_eq!(opts.min_duration(), Duration::from_millis(750));
    assert!(!opts.initial_cut());

    // with_min_frames alternate.
    let opts_frames = Options::default().with_min_frames(15, fps30);
    assert_eq!(opts_frames.min_duration(), Duration::from_millis(500));

    // In-place setters, chainable.
    let mut opts = Options::default();
    opts
      .set_threshold(100)
      .set_method(Method::Floor)
      .set_fade_bias(-0.5)
      .set_add_final_scene(true)
      .set_min_duration(Duration::from_secs(2))
      .set_initial_cut(true);
    assert_eq!(opts.threshold(), 100);
    assert_eq!(opts.method(), Method::Floor);
    assert_eq!(opts.fade_bias(), -0.5);
    assert!(opts.add_final_scene());
    assert!(opts.initial_cut());

    opts.set_min_frames(60, fps30);
    assert_eq!(opts.min_duration(), Duration::from_secs(2));
  }

  #[test]
  fn detector_options_accessor() {
    let opts = Options::default().with_threshold(77);
    let det = Detector::new(opts);
    assert_eq!(det.options().threshold(), 77);
  }

  #[test]
  fn initial_cut_false_seeds_last_cut_at_ts() {
    // With `initial_cut = false`, the first frame should seed
    // `last_scene_cut` to the frame's own ts (not ts - min_duration), so
    // the first complete fade-in-from-out transition that happens within
    // min_duration of the first frame is suppressed. This exercises the
    // `else` branch of the seed in process_with_mean.
    let opts = Options::default()
      .with_min_duration(Duration::from_millis(200))
      .with_initial_cut(false);
    let mut det = Detector::new(opts);
    let bright = uniform_luma(200, 0);
    let dark = uniform_luma(5, 0);

    // A full fade cycle compressed into 200 ms — the emitted cut's placed
    // midpoint is too close to the seeded ts=0 anchor → gate fails.
    det.process_luma(luma(&bright, 8, 8, 0));
    det.process_luma(luma(&dark, 8, 8, 50));
    let cut = det.process_luma(luma(&bright, 8, 8, 150));
    assert!(
      cut.is_none(),
      "cut should be suppressed with initial_cut=false"
    );
  }
}
