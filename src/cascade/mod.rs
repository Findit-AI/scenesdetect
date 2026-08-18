//! Integrated cascade detector: warm per-frame cut lanes → cheap
//! keyframe gates → expensive keyframe metrics, composed into one
//! sans-I/O state machine.
//!
//! # Workflow
//!
//! Per pushed frame, in order:
//!
//! 1. **Cut lanes** — every enabled lane runs warm on every frame at
//!    its standalone configuration and cuts directly: [`threshold`]
//!    (mean-luma fades), [`histogram`] over the luma plane, optional
//!    [`histogram`] instances over the hue and saturation planes (the
//!    chroma guards: transitions invisible to every luma-derived
//!    signal still shift the chroma distributions), [`phash`],
//!    [`content`], and [`adaptive`]. Direct histogram / phash cuts are
//!    attributed via [`Provenance::Histogram`] / [`Provenance::Phash`].
//! 2. **Contention** — reports contend on timestamp each push: the
//!    earliest admissible closes the shot, and strictly-later
//!    same-push losers are deferred to re-contend on later pushes (an
//!    [`adaptive`] verdict lags by its window, so same-push collisions
//!    are normal). The forced split ([`Options::max_scene_duration`])
//!    contends as the lowest-precedence candidate.
//! 3. **Keyframe selection (per window)** — every frame's metrics feed
//!    an internal [`select::Detector`]. Cheap metrics (luma mean /
//!    variance, saturation variance, clipping) run first; only frames
//!    passing the cheap gates pay for the expensive metrics (sharpness,
//!    noise, motion blur, colorfulness). The selector's buffer is
//!    drained one **window** at a time — the window equals
//!    [`target_interval`](select::Options::target_interval) (default
//!    1 s) — as each window closes or a cut confirms: the window's
//!    winner is emitted as a [`Event::Keyframe`] carrying
//!    [`Provenance::Quality`] (passed every strict gate) or
//!    [`Provenance::Fallback`] (least-bad pick). A consumer therefore
//!    caches only a small bounded span of frames, and keyframes associate to
//!    scenes by timestamp (a keyframe's scene is the [`Event::Scene`]
//!    whose range contains its timestamp).
//!
//! Per-frame cost is every enabled lane's pass over the frame plus the
//! cheap keyframe metrics; the expensive keyframe metrics run only for
//! frames that pass the cheap gates.
//!
//! The lanes are pure signal generators: every lane runs with its
//! internal `min_duration` zeroed, and the machine owns all timeline
//! policy — per-lane spacing and warm-up applied via the machine's
//! admissibility gates — so a report the machine drops can never
//! advance a lane's private debounce anchor and shadow a later valid
//! cut. One documented trade-off: [`content::FilterMode::Merge`]
//! re-times flash runs via `min_duration`, and that re-timing is not
//! reproduced inside the cascade — the machine's spacing enforces the
//! same cut cadence without it.
//!
//! # Sans-I/O contract
//!
//! [`Detector::push`] is a pure state transition returning at most one
//! [`Event`]; the machine never blocks, spawns, or calls out.
//! [`Detector::finalize`] closes the open shot, returns every remaining
//! output in order, and refreshes all state for the next stream;
//! [`Detector::clear`] refreshes without emitting. One stream per
//! detector — multiplexing belongs to the caller, as does pairing
//! keyframe timestamps back to pixel data.
//!
//! Event invariants:
//!
//! - Every [`Event::Keyframe`] of a shot is delivered before that
//!   shot's [`Event::Scene`]: the window up to a confirming cut is
//!   finalized first, so its keyframes enter the queue ahead of the
//!   scene. Keyframe timestamps are strictly increasing within a shot.
//! - Keyframes associate to scenes by timestamp alone: a keyframe's
//!   scene is the [`Event::Scene`] whose half-open range contains its
//!   timestamp. This stays total even around lagged boundaries, where
//!   a keyframe surfacing near a boundary may belong to the *next*
//!   scene (with sparse pushes it can even be delivered before the
//!   *prior* scene's output; the partition keeps association total
//!   regardless of arrival order).
//! - Scene ranges are half-open `[start, end)`, all emitted in the
//!   stream's canonical timebase (the first frame's), with both
//!   endpoints ceiling-rescaled through the same function — so
//!   consecutive ranges stay exactly adjacent AND the emitted ranges
//!   tile the stream, keeping timestamp association total even under
//!   mixed-timebase pushes (a sub-tick scene collapses and is skipped;
//!   its keyframes parent to the neighboring range).
//! - The look-back / caching bound is **one window plus the adaptive
//!   lag horizon**: the selector holds about one
//!   [`target_interval`](select::Options::target_interval) of frames
//!   (a few more while an adaptive lane's lagged cut could still
//!   reclaim them), so a consumer pairing keyframe timestamps back to
//!   pixels caches only a small, bounded span.
//! - Emission of a closed window's winner (and of a lag-refined scene)
//!   happens on the call *after* the deciding frame: outputs describe
//!   earlier frames, never the future.
//! - The undelivered backlog is bounded: configurations that confirm
//!   boundaries faster than one output per push shed their OLDEST
//!   queued keyframes past a documented cap; scenes are never shed.
//! - Concurrent boundary sources (the fade, lane reports, deferred
//!   survivors, the forced split) contend on timestamp each push: the
//!   earliest admissible closes, strictly-later losers re-contend on
//!   later pushes. The one residual: fade cuts from the terminal
//!   threshold lane interpolate between frames and may lag a fade's
//!   full length; such a cut never coincides with a pushed frame and
//!   timestamp association stays total, but a dark-stretch fallback
//!   pick near a long fade can end up attributed to the post-fade
//!   scene.
//!
//! # Trade-offs, stated plainly
//!
//! - A transition below every enabled lane's threshold is missed by
//!   design.
//! - Keyframes are selected per fixed-width window rather than over the
//!   whole (known-length) shot: a shot spanning several windows yields
//!   roughly one keyframe per window, and the per-shot adaptive
//!   sharpness floor never engages within a single short window (it
//!   falls back to the static
//!   [`min_sharpness`](select::Options::min_sharpness)). The trade is a
//!   bounded one-window caching cost against whole-shot optimality.
//! - Detectors here, as everywhere in this crate, expect downscaled
//!   analysis frames (the PySceneDetect 200–400 px convention).

use core::{mem, time::Duration};

use std::{collections::VecDeque, vec::Vec};

use bitflags::bitflags;
use derive_more::{Display, IsVariant};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
  adaptive, content,
  frame::{ChannelOrder, HsvFrame, LumaFrame, RgbFrame, TimeRange, Timebase, Timestamp},
  histogram,
  keyframe::{
    clipping, colorfulness, luma,
    metrics::FrameMetrics,
    motion_blur, noise, saturation,
    select::{self, hard_gate},
    sharpness,
  },
  phash, threshold,
};

/// Extra slack on the adaptive lag horizon, absorbing report jitter
/// around a boundary.
const RING_SLACK: usize = 2;

/// Upper bound on the pixel area of one analysis frame — a per-push
/// compute bound: every enabled lane scans the frame each push, so
/// frame size is capped up front; the crate's downscale convention (a
/// few hundred pixels per side) sits orders of magnitude below this
/// cap.
const MAX_FRAME_PIXELS: u64 = 1 << 20;

/// Upper bound on the adaptive lag horizon: it caps the
/// `2 * window + 1 + RING_SLACK` span by which an [`adaptive`] verdict
/// can lag the current frame — a window wide enough to exceed this is a
/// configuration error, so [`Detector::try_new`] rejects it.
const MAX_RING: usize = 64;

/// Upper bound on the undelivered output backlog. `push` returns at
/// most one output per call, so a configuration that confirms a
/// boundary on nearly every push (zero detector thresholds and
/// spacings are type-valid) produces outputs faster than they drain.
/// Past this cap the OLDEST queued keyframe is shed — never a scene:
/// scenes are the timeline and enter at most one per push (net zero
/// against the drain), while keyframes are samples and degrade
/// gracefully. Realistic configurations never approach the cap.
const MAX_PENDING_OUTPUTS: usize = 256;

/// Upper bound on a histogram lane's bin count. Every enabled lane
/// eagerly allocates three bin-sized buffers at construction, so an
/// unbounded count is a configuration error surfaced as
/// [`OptionsError`], not an allocation attempt; 8-bit planes occupy at
/// most 256 bins, so the cap is already generous.
const MAX_HISTOGRAM_BINS: usize = 1 << 16;

/// Upper bound on the pHash working size (`size * lowpass`): the lane
/// allocates DCT bases and a resized image quadratic in it, so an
/// unbounded value is a configuration error surfaced as
/// [`OptionsError`], not an allocation attempt. The standalone default
/// is 96; the cap is already generous.
const MAX_PHASH_IMSIZE: u64 = 1 << 10;

/// Upper bound on an explicit content/adaptive edge-detection kernel:
/// the Canny dilation path scans kernel-sized windows per pixel, so an
/// unbounded explicit kernel is a first-push CPU hazard surfaced as
/// [`OptionsError`] instead. The auto kernel never exceeds this.
const MAX_EDGE_KERNEL: u32 = 1 << 8;

// ---------------------------------------------------------------------------
// Frames — the validated per-frame input bundle
// ---------------------------------------------------------------------------

/// The three views of one analysis frame, validated once to share
/// dimensions and timestamp. Each view is already stride-aware; the
/// detector never asks why the caller chose this resolution.
#[derive(Debug, Clone, Copy)]
pub struct Frames<'a> {
  rgb: RgbFrame<'a>,
  luma: LumaFrame<'a>,
  hsv: HsvFrame<'a>,
}

impl<'a> Frames<'a> {
  /// Bundles the three views, rejecting mismatched dimensions or
  /// timestamps so the cascade never computes on inconsistent planes.
  pub fn try_new(
    rgb: RgbFrame<'a>,
    luma: LumaFrame<'a>,
    hsv: HsvFrame<'a>,
  ) -> Result<Self, FramesError> {
    let dims = (rgb.width(), rgb.height());
    if dims.0 == 0 || dims.1 == 0 {
      return Err(FramesError::ZeroDimensions);
    }
    let pixels = u64::from(dims.0) * u64::from(dims.1);
    if pixels > MAX_FRAME_PIXELS {
      return Err(FramesError::FrameTooLarge { pixels });
    }
    // Every view carries its own timestamp, and the cut detectors
    // report whichever one they were fed — so all three timebases must
    // be sound, not just RGB's (a zero-numerator timebase makes every
    // pts the same instant, and a boundary range built from it would
    // need a rescale that panics downstream). Checked before semantic
    // equality, which would otherwise let a degenerate semantic-zero
    // view compare equal to a healthy one.
    if rgb.timestamp().timebase().num() == 0
      || luma.timestamp().timebase().num() == 0
      || hsv.timestamp().timebase().num() == 0
    {
      return Err(FramesError::ZeroTimebase);
    }
    if (luma.width(), luma.height()) != dims || (hsv.width(), hsv.height()) != dims {
      return Err(FramesError::DimensionMismatch);
    }
    if rgb.timestamp().cmp_semantic(&luma.timestamp()).is_ne()
      || rgb.timestamp().cmp_semantic(&hsv.timestamp()).is_ne()
    {
      return Err(FramesError::TimestampMismatch);
    }
    Ok(Self { rgb, luma, hsv })
  }

  /// The packed-RGB view.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> RgbFrame<'a> {
    self.rgb
  }

  /// The luma view.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma(&self) -> LumaFrame<'a> {
    self.luma
  }

  /// The HSV view.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hsv(&self) -> HsvFrame<'a> {
    self.hsv
  }

  /// The frame's presentation timestamp (shared by all three views).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timestamp(&self) -> Timestamp {
    self.rgb.timestamp()
  }
}

/// Why a [`Frames`] bundle was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum FramesError {
  /// The three views disagree on width or height.
  #[error("the rgb / luma / hsv views disagree on dimensions")]
  DimensionMismatch,
  /// The three views disagree on the presentation timestamp.
  #[error("the rgb / luma / hsv views disagree on timestamp")]
  TimestampMismatch,
  /// The views are zero-sized; there is nothing to analyze (and the
  /// downstream reductions assume at least one pixel).
  #[error("the views have zero width or height")]
  ZeroDimensions,
  /// The views exceed `MAX_FRAME_PIXELS`; per-push compute is
  /// bounded up front (analysis frames per the crate's downscale
  /// convention sit far below this).
  #[error("the views carry {pixels} pixels, past the cap")]
  FrameTooLarge {
    /// The offending `width * height` product.
    pixels: u64,
  },
  /// The timestamp timebase has a zero numerator: every pts would
  /// name the same instant, and range arithmetic would panic.
  #[error("the timestamp timebase numerator is zero")]
  ZeroTimebase,
}

// ---------------------------------------------------------------------------
// Provenance — which path produced an output, with its computed scores
// ---------------------------------------------------------------------------

/// Which plane a histogram lane watched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum HistogramPlane {
  /// The luma plane.
  Luma,
  /// The hue plane.
  Hue,
  /// The saturation plane.
  Saturation,
}

impl HistogramPlane {
  /// Stable snake_case slug — the canonical string form of every
  /// variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Luma => "luma",
      Self::Hue => "hue",
      Self::Saturation => "saturation",
    }
  }
}

/// Computed values behind a [`Provenance::Histogram`] boundary (a
/// histogram lane cutting directly).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistogramScore {
  plane: HistogramPlane,
  correlation: f64,
}

impl HistogramScore {
  /// Which plane's distribution shifted.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn plane(&self) -> HistogramPlane {
    self.plane
  }

  /// The correlation between the two frames' histograms (`1.0` =
  /// identical shape; the lane fires when it falls below its
  /// configured threshold).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn correlation(&self) -> f64 {
    self.correlation
  }
}

/// Computed values behind a [`Provenance::Phash`] boundary (the
/// perceptual-hash lane cutting directly).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhashScore {
  distance: f64,
}

impl PhashScore {
  /// The normalised Hamming distance that fired the lane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn distance(&self) -> f64 {
    self.distance
  }
}

/// Computed values behind a [`Provenance::Threshold`] boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdScore {
  mean: f64,
  floor: u8,
}

impl ThresholdScore {
  /// Mean luma of the frame that fired the fade lane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn mean(&self) -> f64 {
    self.mean
  }

  /// The configured fade threshold it crossed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn floor(&self) -> u8 {
    self.floor
  }
}

/// Computed values behind a [`Provenance::Content`] boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentScore {
  score: f64,
  threshold: f64,
  components: content::Components,
}

impl ContentScore {
  /// The weighted content score at the boundary.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn score(&self) -> f64 {
    self.score
  }

  /// The configured content threshold it exceeded.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold(&self) -> f64 {
    self.threshold
  }

  /// The per-channel component deltas behind the score.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn components(&self) -> content::Components {
    self.components
  }
}

/// Computed values behind a [`Provenance::Adaptive`] boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveScore {
  ratio: f64,
  score: f64,
}

impl AdaptiveScore {
  /// The adaptive ratio (content score over the local window average)
  /// at the boundary.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn ratio(&self) -> f64 {
    self.ratio
  }

  /// The raw content score at the boundary.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn score(&self) -> f64 {
    self.score
  }
}

/// Computed values behind a [`Provenance::MaxSpan`] boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanScore {
  span: Duration,
  cap: Duration,
}

impl SpanScore {
  /// The open shot's span when the split fired.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn span(&self) -> Duration {
    self.span
  }

  /// The configured [`Options::max_scene_duration`] cap it exceeded.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn cap(&self) -> Duration {
    self.cap
  }
}

/// What produced an output, carrying the calculated information behind
/// the decision (detector scores for scenes, quality metrics for
/// keyframes).
#[derive(Debug, Clone, PartialEq, IsVariant)]
#[non_exhaustive]
pub enum Provenance {
  /// The terminal fade lane cut directly.
  Threshold(ThresholdScore),
  /// A histogram lane cut directly.
  Histogram(HistogramScore),
  /// The perceptual-hash lane cut directly.
  Phash(PhashScore),
  /// A [`content`] boundary.
  Content(ContentScore),
  /// An [`adaptive`] boundary (window-lagged).
  Adaptive(AdaptiveScore),
  /// The forced split: no detector fired within
  /// [`Options::max_scene_duration`].
  MaxSpan(SpanScore),
  /// The shot was closed by [`Detector::finalize`] (end of stream).
  Finalized,
  /// A keyframe that passed every quality gate; the payload is its
  /// full computed metrics.
  Quality(FrameMetrics),
  /// A keyframe from an interval where no frame passed the gates: the
  /// least-bad frame, surfaced so the interval still has coverage. The
  /// payload holds the metrics that were computed (cheap-gate failures
  /// skip the expensive metrics, which read as zero).
  Fallback(FrameMetrics),
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// One selected keyframe of the open shot.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeEvent {
  ts: Timestamp,
  provenance: Provenance,
}

impl KeyframeEvent {
  /// The selected frame's presentation timestamp.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn timestamp(&self) -> Timestamp {
    self.ts
  }

  /// The selection path and its computed metrics.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn provenance_ref(&self) -> &Provenance {
    &self.provenance
  }

  /// Decompose for the consumer: `(timestamp, provenance)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn into_parts(self) -> (Timestamp, Provenance) {
    (self.ts, self.provenance)
  }
}

/// One closed shot: its half-open range and what closed it. Keyframes
/// are not aggregated here — the caller collects the [`Event::Keyframe`]
/// stream (see the module docs for the association rules).
#[derive(Debug, Clone, PartialEq)]
pub struct SceneEvent {
  range: TimeRange,
  provenance: Provenance,
}

impl SceneEvent {
  /// The shot's half-open pts range `[start, end)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn range(&self) -> TimeRange {
    self.range
  }

  /// What closed the shot, with its computed scores.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn provenance_ref(&self) -> &Provenance {
    &self.provenance
  }

  /// Decompose for the consumer: `(range, provenance)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn into_parts(self) -> (TimeRange, Provenance) {
    (self.range, self.provenance)
  }
}

/// One emitted decision: the pushed stream is classified into scenes
/// and keyframes; everything else is silence (`None` from
/// [`Detector::push`]).
#[derive(Debug, Clone, PartialEq, IsVariant)]
#[non_exhaustive]
pub enum Event {
  /// A selected keyframe of the open shot.
  Keyframe(KeyframeEvent),
  /// A closed shot.
  Scene(SceneEvent),
}

// ---------------------------------------------------------------------------
// Lane selection — which detectors / extractors run
// ---------------------------------------------------------------------------

bitflags! {
  /// Which cut lanes run — every enabled lane runs on every frame and
  /// cuts directly. Enablement lives here; *how* each enabled lane
  /// behaves lives in its [`Options`] field — the two concerns never
  /// share a knob.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub struct Detectors: u8 {
    /// The mean-luma fade lane.
    const THRESHOLD = 1;
    /// The luma-plane histogram lane.
    const LUMA_HISTOGRAM = 1 << 1;
    /// The hue-plane histogram lane (hue-rotation guard).
    const HUE_HISTOGRAM = 1 << 2;
    /// The saturation-plane histogram lane (chroma guard).
    const SATURATION_HISTOGRAM = 1 << 3;
    /// The perceptual-hash lane.
    const PHASH = 1 << 4;
    /// The [`content`] lane.
    const CONTENT = 1 << 5;
    /// The [`adaptive`] lane (its verdicts lag by the window width).
    const ADAPTIVE = 1 << 6;
  }
}

impl Default for Detectors {
  /// The module-doc default: every lane except the hue-plane
  /// histogram.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::THRESHOLD
      | Self::LUMA_HISTOGRAM
      | Self::SATURATION_HISTOGRAM
      | Self::PHASH
      | Self::CONTENT
      | Self::ADAPTIVE
  }
}

// Bitflags serialize as their human-readable flag strings (for
// example "THRESHOLD | CONTENT"), the bitflags-recommended serde
// form: stable across bit reassignments and editable in config files.
#[cfg(feature = "serde")]
impl Serialize for Detectors {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    bitflags::serde::serialize(self, serializer)
  }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Detectors {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    bitflags::serde::deserialize(deserializer)
  }
}

bitflags! {
  /// Which expensive (stage-4) keyframe metrics run for gate-passing
  /// frames. The cheap stage-3 metrics (luma mean / variance,
  /// saturation variance, clipping) always run — they feed the gates
  /// and cost a few reduce passes. A disabled extractor's metric reads
  /// as zero in [`FrameMetrics`] and in the composite, and its
  /// dedicated gate is skipped (disabling `SHARPNESS` skips the
  /// `min_sharpness` strictness check; disabling `MOTION_BLUR`
  /// effectively disarms the motion-blur gate).
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub struct Extractors: u8 {
    /// Tenengrad sharpness.
    const SHARPNESS = 1;
    /// Immerkaer noise estimate.
    const NOISE = 1 << 1;
    /// Directional-gradient motion blur.
    const MOTION_BLUR = 1 << 2;
    /// Hasler-Süsstrunk colorfulness.
    const COLORFULNESS = 1 << 3;
  }
}

impl Default for Extractors {
  /// All stage-4 extractors enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::all()
  }
}

#[cfg(feature = "serde")]
impl Serialize for Extractors {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    bitflags::serde::serialize(self, serializer)
  }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Extractors {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    bitflags::serde::deserialize(deserializer)
  }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Configuration of the cascade: which lanes run ([`Detectors`] /
/// [`Extractors`]), how each enabled lane behaves (its embedded
/// options), the keyframe policy source, and the structural knobs.
/// Defaults follow the module docs: every lane enabled except the
/// hue-plane histogram, each at its standalone defaults.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  detectors: Detectors,
  extractors: Extractors,
  threshold: threshold::Options,
  luma_histogram: histogram::Options,
  hue_histogram: histogram::Options,
  sat_histogram: histogram::Options,
  phash: phash::Options,
  content: content::Options,
  adaptive: adaptive::Options,
  select: select::Options,
  #[cfg_attr(feature = "serde", serde(with = "humantime_serde"))]
  max_scene_duration: Option<Duration>,
  channel_order: ChannelOrder,
}

/// Default [`Options::max_scene_duration`]: a shot in which no
/// detector fires for this long is force-closed.
pub const DEFAULT_MAX_SCENE_DURATION: Duration = Duration::from_secs(15);

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

/// Every histogram lane runs with its internal min_duration zeroed —
/// the machine owns timeline policy (see [`Detector::try_new`]), and a
/// lane that self-debounces is poisoned by reports the machine drops.
fn signal_lane(options: histogram::Options) -> histogram::Options {
  options.with_min_duration(Duration::ZERO)
}

impl Options {
  /// Defaults per the module docs: every lane in [`Detectors::default`]
  /// at its standalone defaults, every stage-4 extractor on, a 1 s
  /// keyframe window, forced split at
  /// [`DEFAULT_MAX_SCENE_DURATION`].
  pub fn new() -> Self {
    Self {
      detectors: Detectors::default(),
      extractors: Extractors::default(),
      threshold: threshold::Options::new(),
      luma_histogram: histogram::Options::new(),
      hue_histogram: histogram::Options::new(),
      sat_histogram: histogram::Options::new(),
      phash: phash::Options::new(),
      content: content::Options::new(),
      adaptive: adaptive::Options::new(),
      // target_interval doubles as the keyframe emission window: the
      // cascade finalizes the selector once per interval (default 1 s)
      // rather than once per known-length shot.
      select: select::Options::new().with_target_interval(Duration::from_secs(1)),
      max_scene_duration: Some(DEFAULT_MAX_SCENE_DURATION),
      channel_order: ChannelOrder::Bgr,
    }
  }

  /// Which cut lanes run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn detectors(&self) -> Detectors {
    self.detectors
  }

  /// Choose which cut lanes run (configuration of each lane stays in
  /// its own options field).
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_detectors(mut self, v: Detectors) -> Self {
    self.detectors = v;
    self
  }

  /// Which stage-4 keyframe extractors run.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn extractors(&self) -> Extractors {
    self.extractors
  }

  /// Choose which stage-4 keyframe extractors run.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_extractors(mut self, v: Extractors) -> Self {
    self.extractors = v;
    self
  }

  /// The terminal fade lane's configuration (enabled via
  /// [`Detectors::THRESHOLD`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold_ref(&self) -> &threshold::Options {
    &self.threshold
  }

  /// Replace the terminal fade lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_threshold(mut self, v: threshold::Options) -> Self {
    self.threshold = v;
    self
  }

  /// The luma-plane histogram lane's configuration (enabled via
  /// [`Detectors::LUMA_HISTOGRAM`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_histogram_ref(&self) -> &histogram::Options {
    &self.luma_histogram
  }

  /// Replace the luma-plane histogram lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_histogram(mut self, v: histogram::Options) -> Self {
    self.luma_histogram = v;
    self
  }

  /// The hue-plane histogram lane's configuration (enabled via
  /// [`Detectors::HUE_HISTOGRAM`] — off by default; enable it to also
  /// guard hue-rotation transitions that keep both luma and saturation
  /// distributions stable).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hue_histogram_ref(&self) -> &histogram::Options {
    &self.hue_histogram
  }

  /// Replace the hue-plane histogram lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_hue_histogram(mut self, v: histogram::Options) -> Self {
    self.hue_histogram = v;
    self
  }

  /// The saturation-plane histogram lane's configuration (enabled
  /// via [`Detectors::SATURATION_HISTOGRAM`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sat_histogram_ref(&self) -> &histogram::Options {
    &self.sat_histogram
  }

  /// Replace the saturation-plane histogram lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_sat_histogram(mut self, v: histogram::Options) -> Self {
    self.sat_histogram = v;
    self
  }

  /// The perceptual-hash lane's configuration (enabled via
  /// [`Detectors::PHASH`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn phash_ref(&self) -> &phash::Options {
    &self.phash
  }

  /// Replace the perceptual-hash lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_phash(mut self, v: phash::Options) -> Self {
    self.phash = v;
    self
  }

  /// The [`content`] lane's options (enabled via
  /// [`Detectors::CONTENT`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn content_ref(&self) -> &content::Options {
    &self.content
  }

  /// Replace the [`content`] lane's options.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_content(mut self, v: content::Options) -> Self {
    self.content = v;
    self
  }

  /// The [`adaptive`] lane's configuration (enabled via
  /// [`Detectors::ADAPTIVE`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn adaptive_ref(&self) -> &adaptive::Options {
    &self.adaptive
  }

  /// Replace the [`adaptive`] lane's configuration.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_adaptive(mut self, v: adaptive::Options) -> Self {
    self.adaptive = v;
    self
  }

  /// The keyframe-policy source: gate thresholds, composite weights,
  /// sharpness floor, and the per-shot cap all come from these
  /// [`select::Options`] (single source of truth with the standalone
  /// selector the cascade drives internally).
  ///
  /// [`target_interval`](select::Options::target_interval) doubles as
  /// the keyframe **emission window**: the cascade finalizes the
  /// selector once per interval (emitting that window's winner) rather
  /// than once per known-length shot, so a consumer caches at most one
  /// window of frames. Defaults to 1 s.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn select_ref(&self) -> &select::Options {
    &self.select
  }

  /// Replace the keyframe-policy source.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_select(mut self, v: select::Options) -> Self {
    self.select = v;
    self
  }

  /// The forced-split cap: a shot in which no detector fires for this
  /// long is closed at the current frame, attributed
  /// [`Provenance::MaxSpan`] (`None` = never force). Defaults to
  /// [`DEFAULT_MAX_SCENE_DURATION`]. The cap is frame-granular: the
  /// shot closes at the first frame at or past the cap, so with sparse
  /// timestamps a scene can exceed the cap by up to one inter-frame
  /// gap (the trailing shot closed by [`Detector::finalize`] stays
  /// under the cap plus one tick).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_scene_duration(&self) -> Option<Duration> {
    self.max_scene_duration
  }

  /// Set the forced-split cap.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_scene_duration(mut self, v: Option<Duration>) -> Self {
    self.max_scene_duration = v;
    self
  }

  /// Byte order of the packed RGB view (consumed by the clipping and
  /// colorfulness metrics).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn channel_order(&self) -> ChannelOrder {
    self.channel_order
  }

  /// Set the packed-RGB byte order.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_channel_order(mut self, v: ChannelOrder) -> Self {
    self.channel_order = v;
    self
  }
}

/// Why a cascade [`Detector`] could not be built.
#[derive(Debug, Clone, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum OptionsError {
  /// No lane enabled: the detector could never close a shot except by
  /// `max_scene_duration`, which is not a detector.
  #[error("no lane enabled")]
  NoCutLane,
  /// The [`adaptive`] lane's options were rejected.
  #[error("adaptive lane rejected: {0}")]
  Adaptive(#[from] adaptive::Error),
  /// The [`content`] lane's options were rejected.
  #[error("content lane rejected: {0}")]
  Content(#[from] content::Error),
  /// One of the enabled histogram lanes' options was rejected.
  #[error("histogram lane rejected: {0}")]
  Histogram(#[from] histogram::Error),
  /// The perceptual-hash lane's options were rejected.
  #[error("phash lane rejected: {0}")]
  Phash(#[from] phash::Error),
  /// The adaptive window would push the lag horizon past `MAX_RING`;
  /// a window that wide is a configuration error.
  #[error("adaptive window_width ({0}) puts the lag horizon past the cap")]
  WindowTooWide(u32),
  /// An enabled histogram lane's bin count exceeds
  /// `MAX_HISTOGRAM_BINS`; each lane eagerly allocates bin-sized
  /// buffers, so an unbounded count is rejected rather than attempted.
  #[error("histogram bins ({0}) exceeds the cap")]
  HistogramBinsTooLarge(usize),
  /// The enabled pHash lane's working size (`size * lowpass`) exceeds
  /// `MAX_PHASH_IMSIZE`; its allocations are quadratic in that size,
  /// so an unbounded value is rejected rather than attempted.
  #[error("phash working size ({0}) exceeds the cap")]
  PhashSizeTooLarge(u64),
  /// A content or adaptive lane combines a nonzero edge weight with an
  /// explicit kernel above `MAX_EDGE_KERNEL`; edge extraction scans
  /// kernel-sized windows per pixel, so an unbounded kernel is
  /// rejected rather than run.
  #[error("edge kernel ({0}) exceeds the cap")]
  KernelTooLarge(u32),
}

/// Why [`Detector::push`] rejected a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant, Error)]
#[non_exhaustive]
pub enum PushError {
  /// The frame's timestamp precedes the stream position: every
  /// downstream invariant (lane recency, window-finalize anchoring,
  /// canonical ranges) assumes a non-decreasing stream, so a regressing frame
  /// is rejected with all state untouched. Equal timestamps are
  /// accepted.
  #[error("frame timestamp regressed behind the stream position")]
  OutOfOrder,
  /// The frame's pts sits at the representable limit (`i64::MAX`), or
  /// its one-tick-past close cannot be represented in the stream's
  /// canonical timebase: the trailing close ends one tick past the
  /// last frame, so such a frame could never be associated with a
  /// scene. Rejected with all state untouched.
  #[error("frame timestamp not representable for scene association")]
  TimestampAtLimit,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// The integrated cascade detector. See the module docs for the
/// workflow and the output contract; construct with
/// [`Detector::try_new`], drive with [`Detector::push`], and close each
/// stream with [`Detector::finalize`].
pub struct Detector {
  options: Options,
  // Terminal fade lane + per-frame lanes (always-on, persistent
  // state).
  threshold: Option<threshold::Detector>,
  luma_hist: Option<histogram::Detector>,
  hue_hist: Option<histogram::Detector>,
  sat_hist: Option<histogram::Detector>,
  phash: Option<phash::Detector>,
  // Keyframe metric detectors (persistent scratch).
  luma_stats: luma::Detector,
  saturation: saturation::Detector,
  clipping: clipping::Detector,
  sharpness: sharpness::Detector,
  noise: noise::Detector,
  motion_blur: motion_blur::Detector,
  colorfulness: colorfulness::Detector,
  // Continuous content / adaptive lane instances.
  cont_content: Option<content::Detector>,
  cont_adaptive: Option<adaptive::Detector>,
  // Visual stream state.
  /// Boundaries that lost a same-push earliest-selection: re-gated
  /// and re-contended on later pushes (bounded by the lane count;
  /// their lanes consumed the transition evidence, so dropping them
  /// would lose the boundary).
  deferred_direct: Vec<(Timestamp, Provenance)>,
  last_confirmed: Option<Timestamp>,
  dims: Option<(u32, u32)>,
  // Shot + interval state.
  shot_start: Option<Timestamp>,
  /// The stream's canonical timebase — the first frame's. Every
  /// emitted [`TimeRange`] rescales into it, so ranges stay adjacent
  /// (a partition) even when later pushes carry other timebases.
  stream_timebase: Option<Timebase>,
  /// Anchor for first-cut warm-up gating (`initial_cut(false)` lanes):
  /// the first frame of the live visual stream — re-seeded on a
  /// dimension reset, because a lane whose state was just cleared must
  /// warm up against the new stream, not the old timeline.
  gate_anchor: Option<Timestamp>,
  last_ts: Option<Timestamp>,
  // Keyframe selection. Every frame's metrics feed `selector`; its
  // buffer is drained one window at a time (see `finalize_window`).
  selector: select::Detector,
  /// Start of the current keyframe window: the lower bound of the
  /// `[last_finalize, point)` range handed to the selector on the next
  /// finalize. Anchored at the first frame, advanced past each closed
  /// window and re-anchored at every confirmed cut.
  last_finalize: Option<Timestamp>,
  /// Frames a confirmed cut can lag behind the current frame (the
  /// adaptive window's span; 0 with no adaptive lane). The periodic
  /// window finalize holds back this many recent frames so a late cut
  /// can still reclaim them for the new scene.
  lag_frames: usize,
  /// The last `lag_frames` frame timestamps; the front is the safe
  /// finalize horizon — no not-yet-confirmed cut can name a target
  /// before it.
  recent_ts: VecDeque<Timestamp>,
  // Event queue (drained one item per call).
  pending: VecDeque<Event>,
}

impl<'a> Frames<'a> {
  /// The same views restamped: the cascade normalizes every frame to
  /// the stream's canonical timebase at admission, so the lanes only
  /// ever observe ONE timebase and no lane-internal rescale can
  /// saturate.
  fn with_canonical_timestamp(&self, ts: Timestamp) -> Frames<'a> {
    let l = self.luma;
    let h = self.hsv;
    let r = self.rgb;
    Frames {
      rgb: RgbFrame::new(r.data(), r.width(), r.height(), r.stride(), ts),
      luma: LumaFrame::new(l.data(), l.width(), l.height(), l.stride(), ts),
      hsv: HsvFrame::new(
        h.hue(),
        h.saturation(),
        h.value(),
        h.width(),
        h.height(),
        h.stride(),
        ts,
      ),
    }
  }
}

impl Detector {
  /// Builds the cascade, validating the composition: at least one
  /// lane, a non-zero target interval, and well-formed lane options.
  pub fn try_new(options: Options) -> Result<Self, OptionsError> {
    let lanes = options.detectors;
    if lanes.is_empty() {
      return Err(OptionsError::NoCutLane);
    }
    // Edge extraction scans kernel-sized windows per pixel: an
    // unbounded explicit kernel must fail validation, not burn the
    // first push.
    for (lane, edges, kernel) in [
      (
        Detectors::CONTENT,
        options.content.weights().delta_edges(),
        options.content.kernel_size(),
      ),
      (
        Detectors::ADAPTIVE,
        options.adaptive.weights().delta_edges(),
        options.adaptive.kernel_size(),
      ),
    ] {
      if lanes.contains(lane)
        && edges != 0.0
        && let Some(k) = kernel
        && k > MAX_EDGE_KERNEL
      {
        return Err(OptionsError::KernelTooLarge(k));
      }
    }
    // The pHash lane allocates quadratically in size * lowpass; an
    // unbounded working size must fail validation, not the allocator.
    if lanes.contains(Detectors::PHASH) {
      let imsize = u64::from(options.phash.size()) * u64::from(options.phash.lowpass());
      if imsize > MAX_PHASH_IMSIZE {
        return Err(OptionsError::PhashSizeTooLarge(imsize));
      }
    }
    // Histogram lanes eagerly allocate three bin-sized buffers each;
    // an unbounded bin count must fail validation, not the allocator.
    for (lane, histogram) in [
      (Detectors::LUMA_HISTOGRAM, &options.luma_histogram),
      (Detectors::HUE_HISTOGRAM, &options.hue_histogram),
      (Detectors::SATURATION_HISTOGRAM, &options.sat_histogram),
    ] {
      if lanes.contains(lane) && histogram.bins() > MAX_HISTOGRAM_BINS {
        return Err(OptionsError::HistogramBinsTooLarge(histogram.bins()));
      }
    }
    // A zero target interval is unconstructible (the select builder
    // asserts), so interval arithmetic is safe by construction.
    // Validate every enabled fallible lane now, so the construction
    // below cannot panic: the lane instances only zero min_duration,
    // which no validity check depends on.
    if lanes.contains(Detectors::ADAPTIVE) {
      adaptive::Detector::try_new(options.adaptive.clone())?;
    }
    if lanes.contains(Detectors::CONTENT) {
      content::Detector::try_new(options.content.clone())?;
    }
    // The adaptive lane lags by its window; that lag bounds how far a
    // confirmed cut can sit behind the current frame, so a window wide
    // enough to push the lag past the cap is rejected as a
    // configuration error.
    let lag_frames = if lanes.contains(Detectors::ADAPTIVE) {
      (options.adaptive.window_width() as usize)
        .checked_mul(2)
        .and_then(|w| w.checked_add(1 + RING_SLACK))
        .filter(|cap| *cap <= MAX_RING)
        .ok_or(OptionsError::WindowTooWide(options.adaptive.window_width()))?
    } else {
      0
    };
    // The selector drives keyframe selection off these options. When
    // the SHARPNESS extractor is disabled, sharpness reads as zero for
    // every frame, so the selector's `min_sharpness` floor would reject
    // every frame from the strict path — the `Extractors` contract says
    // disabling SHARPNESS skips that strictness check, so the floor is
    // lowered to a non-gating finite value here (the selector accepts
    // finite negatives as "floor disabled").
    // Fixed keyframe windows are not shot boundaries, so the selector's
    // first/last-bucket margins (which guard against scene-cut slop)
    // would only shave a window's edges — and a sparse window could
    // then emit nothing. Disable them; the cut side owns boundary slop.
    let select_opts = if options.extractors.contains(Extractors::SHARPNESS) {
      options.select.with_margin_ratio(0.0)
    } else {
      options
        .select
        .with_min_sharpness(-1.0)
        .with_margin_ratio(0.0)
    };
    Ok(Self {
      // EVERY lane runs WARM on every frame with its internal
      // min_duration ZEROED — the machine applies each lane's
      // configured spacing in its admissibility gates, so a
      // machine-REJECTED report cannot advance a lane's own debounce
      // anchor and shadow a later valid cut. Lanes keep signal state
      // only (previous-frame views, rolling windows).
      // Trade-off: content::FilterMode::Merge re-times flash runs via
      // min_duration; that re-timing is not reproduced here — machine
      // spacing enforces the same cadence (see the module docs).
      threshold: lanes.contains(Detectors::THRESHOLD).then(|| {
        threshold::Detector::new(options.threshold.clone().with_min_duration(Duration::ZERO))
      }),
      luma_hist: lanes
        .contains(Detectors::LUMA_HISTOGRAM)
        .then(|| histogram::Detector::try_new(signal_lane(options.luma_histogram.clone())))
        .transpose()?,
      hue_hist: lanes
        .contains(Detectors::HUE_HISTOGRAM)
        .then(|| histogram::Detector::try_new(signal_lane(options.hue_histogram.clone())))
        .transpose()?,
      sat_hist: lanes
        .contains(Detectors::SATURATION_HISTOGRAM)
        .then(|| histogram::Detector::try_new(signal_lane(options.sat_histogram.clone())))
        .transpose()?,
      phash: lanes
        .contains(Detectors::PHASH)
        .then(|| phash::Detector::try_new(options.phash.clone().with_min_duration(Duration::ZERO)))
        .transpose()?,
      cont_content: lanes.contains(Detectors::CONTENT).then(|| {
        content::Detector::try_new(options.content.clone().with_min_duration(Duration::ZERO))
          .expect("validated above")
      }),
      cont_adaptive: lanes.contains(Detectors::ADAPTIVE).then(|| {
        adaptive::Detector::try_new(options.adaptive.clone().with_min_duration(Duration::ZERO))
          .expect("validated above")
      }),
      luma_stats: luma::Detector::new(luma::Options::new()),
      saturation: saturation::Detector::new(saturation::Options::new()),
      clipping: clipping::Detector::new(clipping::Options::new()),
      sharpness: sharpness::Detector::new(sharpness::Options::new()),
      noise: noise::Detector::new(noise::Options::new()),
      motion_blur: motion_blur::Detector::new(motion_blur::Options::new()),
      colorfulness: colorfulness::Detector::new(
        colorfulness::Options::new().with_channel_order(options.channel_order),
      ),
      deferred_direct: Vec::new(),
      last_confirmed: None,
      dims: None,
      shot_start: None,
      stream_timebase: None,
      gate_anchor: None,
      last_ts: None,
      selector: select::Detector::new(select_opts),
      last_finalize: None,
      lag_frames,
      recent_ts: VecDeque::new(),
      pending: VecDeque::new(),
      options,
    })
  }

  /// The configured options.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn options(&self) -> &Options {
    &self.options
  }

  /// One frame through the cascade: a pure state transition returning
  /// at most one output. Timestamps must be non-decreasing within a
  /// stream — a regressing frame is rejected with
  /// [`PushError::OutOfOrder`] and no state change (equal timestamps
  /// are accepted).
  pub fn push(&mut self, frames: Frames<'_>) -> Result<Option<Event>, PushError> {
    let ts = frames.timestamp();
    if ts.pts() == i64::MAX {
      // The trailing close ends one tick PAST the last frame; a frame
      // at the representable limit could never be closed past and
      // would silently fall out of the final scene's half-open range.
      return Err(PushError::TimestampAtLimit);
    }
    let frames = match self.stream_timebase {
      Some(tb) => {
        // Range endpoints are ceiling-rescaled into the canonical
        // (first frame's) timebase: a frame whose one-past close
        // cannot be represented there would collapse the trailing
        // scene and silently drop frames — the canonical-rescale
        // sibling of the raw-limit guard above.
        let one_past = Timestamp::new(ts.pts() + 1, ts.timebase());
        if rescale_ceil_checked(one_past, tb).is_none() {
          return Err(PushError::TimestampAtLimit);
        }
        // Normalize: lanes and shot bookkeeping observe ONE timebase,
        // so no lane-internal rescale (the threshold fade
        // interpolation rescales across frame timebases) can ever
        // saturate. Ceiling keeps the stream monotone.
        frames.with_canonical_timestamp(Timestamp::new(rescale_ceil(ts, tb), tb))
      }
      None => frames,
    };
    let ts = frames.timestamp();
    if ts.pts() == i64::MAX {
      // The canonical ceiling rescale can land on the limit even when
      // the source pts and its one-past were both representable; the
      // trailing close (`last_ts + 1`) would then saturate and orphan
      // this frame's keyframe.
      return Err(PushError::TimestampAtLimit);
    }
    // Ordering is judged in the canonical space the stream actually
    // runs in (ceiling rescale is monotone; distinct fine timestamps
    // may legally become EQUAL coarse ones).
    if self
      .last_ts
      .is_some_and(|last| ts.cmp_semantic(&last).is_lt())
    {
      return Err(PushError::OutOfOrder);
    }
    let dims = (frames.luma().width(), frames.luma().height());
    if self.dims.is_some_and(|d| d != dims) {
      // Cross-size deltas are meaningless: reset the visual state but
      // keep the shot and its candidates (timestamps stay valid). The
      // warm-up anchor re-seeds at this frame: cleared lanes gate
      // their first cut against the new visual stream.
      self.reset_visual();
      self.gate_anchor = Some(ts);
    }
    self.dims = Some(dims);
    self.last_ts = Some(ts);
    if self.gate_anchor.is_none() {
      self.gate_anchor = Some(ts);
    }
    if self.stream_timebase.is_none() {
      self.stream_timebase = Some(ts.timebase());
    }
    if self.shot_start.is_none() {
      self.shot_start = Some(ts);
      // Anchor the first keyframe window at the first frame of the
      // stream so the selector finalizes from a real lower bound.
      if self.last_finalize.is_none() {
        self.last_finalize = Some(ts);
      }
    }

    // Track the recent-frame window first so `safe_horizon` is current
    // for both the overflow recovery below and the periodic finalize.
    if self.lag_frames > 0 {
      self.recent_ts.push_back(ts);
      while self.recent_ts.len() > self.lag_frames {
        self.recent_ts.pop_front();
      }
    }

    // Every frame's metrics feed the selector — including the frame at
    // a direct cut, which `finalize_window(cut)` then leaves buffered
    // (half-open `[., cut)`) for the new scene. BufferFull realistically
    // never triggers because windows flush at most one interval apart.
    // The recovery drain stops at `safe_horizon`, NEVER past it, so a
    // later lagged adaptive cut can still reclaim its frames — the
    // held-back tail is at most `lag_frames` (<= MAX_RING) frames, so
    // draining the safe bulk always frees the buffer. The only residual
    // drop is surplus frames sharing one coarse instant, already
    // represented by that window's keyframe. OutOfOrder cannot happen —
    // the stream is normalized and the PTS guard already ran.
    let metrics = self.frame_metrics(&frames);
    if self.selector.observe(ts, metrics).is_err() {
      let safe = self.safe_horizon(ts).unwrap_or(ts);
      self.finalize_window(safe);
      let _ = self.selector.observe(ts, metrics);
    }

    let window = self.options.select.target_interval();
    let boundary = self.detect_boundary(&frames, ts);
    match boundary {
      Some((cut, provenance)) => {
        // Emit the closing shot's keyframes for `[last_finalize, cut)`
        // BEFORE the scene, then close the scene and re-anchor the
        // window at the cut. The frame at a direct cut (ts == cut)
        // stays buffered for the new scene (half-open range).
        self.finalize_window(cut);
        self.close_shot_at(cut, provenance);
      }
      None => {
        // No cut: finalize the whole windows that no lagged cut can
        // still reclaim, in ONE call. `finalize_shot` buckets the span
        // by `target_interval` internally and clamps the bucket count
        // to `max_frames_per_shot`, so the work is bounded by the
        // buffered frames — never by the number of (possibly empty)
        // windows a sparse timestamp jump spans (a tiny interval plus a
        // large gap must not spin a window-per-iteration loop inside
        // one push). The adaptive lane reports a cut up to `lag_frames`
        // frames late, so frames newer than the safe horizon stay
        // buffered; the held-back tail drains at the next cut or at
        // `finalize`.
        if let Some(safe) = self.safe_horizon(ts)
          && let Some(lf) = self.last_finalize
        {
          // safe and lf share the canonical timebase post-normalization.
          // The saturating rung clamps an absurd window at `i64::MAX` as
          // the old bare `duration_to_pts` did, and panics only on a
          // zero-numerator timebase — which `Frames::try_new` rejects at
          // admission, so `lf` never carries one.
          let window_pts = lf.timebase().saturating_duration_to_pts(window).max(1);
          let elapsed_pts = safe.pts().saturating_sub(lf.pts());
          if elapsed_pts >= window_pts {
            // The last whole-window boundary at or before `safe`: keeps
            // the per-window cadence (a multi-window catch-up
            // sub-buckets into one keyframe per window) while finalizing
            // in a single bounded call.
            let boundary = lf.pts() + (elapsed_pts / window_pts) * window_pts;
            self.finalize_window(Timestamp::new(boundary, lf.timebase()));
          }
        }
      }
    }
    Ok(self.pending.pop_front())
  }

  /// Finalizes the selector for `[last_finalize, point)`, emitting each
  /// winner as a [`Event::Keyframe`], then advances `last_finalize` to
  /// `point`. Frames at or after `point` stay buffered for the next
  /// window or scene.
  /// The latest timestamp it is safe to finalize through: no
  /// not-yet-confirmed cut can name a target before it. With no
  /// lagging lane the current frame is safe; otherwise it is the
  /// timestamp `lag_frames` frames back (`None` while still warming
  /// up, so nothing is finalized periodically yet).
  fn safe_horizon(&self, ts: Timestamp) -> Option<Timestamp> {
    if self.lag_frames == 0 {
      Some(ts)
    } else if self.recent_ts.len() >= self.lag_frames {
      self.recent_ts.front().copied()
    } else {
      None
    }
  }

  fn finalize_window(&mut self, point: Timestamp) {
    let Some(start) = self.last_finalize else {
      return;
    };
    if point.cmp_semantic(&start).is_le() {
      // Backward re-anchor: a long fade can interpolate a cut behind an
      // already-advanced window anchor. The pre-cut frames were already
      // drained by the prior periodic finalize, so there is nothing to
      // emit here; their keyframes parented to the post-fade scene by
      // timestamp (the documented long-fade residual). Do NOT "fix"
      // this into a drain — it would double-count.
      self.last_finalize = Some(point);
      return;
    }
    // start and point share the canonical timebase post-normalization.
    // A periodic finalize hands exactly one window (one
    // `target_interval`) → one keyframe; a finalize at a cut hands the
    // held-back tail, which the selector sub-buckets into one keyframe
    // per `target_interval`.
    if let Some(range) = TimeRange::try_new(start.pts(), point.pts(), point.timebase()) {
      for sel in self.selector.finalize_shot(range) {
        let provenance = if sel.is_strict() {
          Provenance::Quality(sel.metrics())
        } else {
          Provenance::Fallback(sel.metrics())
        };
        self.enqueue(Event::Keyframe(KeyframeEvent {
          ts: sel.timestamp(),
          provenance,
        }));
      }
    }
    self.last_finalize = Some(point);
  }

  /// Closes the open shot — the range ends one tick past the last
  /// pushed frame — returns every remaining output in order, and
  /// refreshes all state for the next stream.
  pub fn finalize(&mut self) -> Vec<Event> {
    // Two sources may still owe boundaries at end of stream: the
    // terminal fade lane (a stream that ends mid fade-out,
    // add_final_scene) and same-push deferred reports, whose lanes
    // already consumed the transition evidence. Close every admissible
    // one in timestamp order through the normal path, finalizing the
    // keyframe window up to each cut first so keyframes precede their
    // scene.
    if let Some(last) = self.last_ts {
      let floor = self.options.threshold.threshold();
      let fade = match &mut self.threshold {
        Some(det) => {
          let mean = det.last_avg().unwrap_or_default();
          det
            .finish(last)
            .map(|cut| (cut, Provenance::Threshold(ThresholdScore { mean, floor })))
        }
        None => None,
      };
      // The EOS fade is freshly OBSERVED here — unlike deferred
      // entries it has not passed an observation-time warm-up gate, so
      // an initial_cut(false) threshold lane applies its warm-up now
      // (the drain below re-checks ordering and spacing only).
      let fade = fade.filter(|(cut, _)| {
        self.options.threshold.initial_cut()
          || self.gate_anchor.is_none_or(|anchor| {
            cut
              .duration_since(&anchor)
              .is_some_and(|d| d >= self.options.threshold.min_duration())
          })
      });
      let mut boundaries = mem::take(&mut self.deferred_direct);
      boundaries.extend(fade);
      boundaries.sort_by(|a, b| a.0.cmp_semantic(&b.0));
      for boundary in boundaries {
        let Some(boundary) = self.revalidate_deferred(boundary) else {
          continue;
        };
        let min_gap = self.lane_gate(&boundary.1);
        if let Some((cut, provenance)) = self.admissible(boundary, min_gap, false) {
          self.finalize_window(cut);
          self.close_shot_at(cut, provenance);
        }
      }
    }
    // Drain the selector's remaining frames as the trailing shot's
    // keyframes, then close that shot one tick past the last frame.
    // The trailing close is gated on the scene actually emitting, so a
    // range that collapses under the canonical timebase cannot leave
    // orphan keyframes (they parent to a neighboring range by
    // timestamp).
    if let (Some(_), Some(last)) = (self.shot_start, self.last_ts) {
      let end = Timestamp::new(last.pts().saturating_add(1), last.timebase());
      self.finalize_window(end);
      self.close_shot_at(end, Provenance::Finalized);
    }
    let outputs = self.pending.drain(..).collect();
    self.reset_all();
    outputs
  }

  /// Discards everything — pending outputs included — and refreshes
  /// for the next stream (the abandon path).
  pub fn clear(&mut self) {
    self.reset_all();
  }

  // --- boundary detection ---------------------------------------------------

  /// Runs every enabled lane on the frame; every lane cuts directly
  /// and the earliest admissible report wins.
  fn detect_boundary(
    &mut self,
    frames: &Frames<'_>,
    ts: Timestamp,
  ) -> Option<(Timestamp, Provenance)> {
    // Admissibility is filtered per candidate BEFORE any
    // earliest-merge: a stale lagged report from one lane must not
    // mask a valid later boundary from another on the same push.
    let shot_start = self.shot_start;
    let last_confirmed = self.last_confirmed;
    // Direct cuts re-apply each lane's configured min_duration against
    // the timeline: a visual reset reseeds the lanes' own debounce
    // state, and without this gate a reseeded lane could cut again
    // inside its spacing window right after a resolution switch.
    let gate_anchor = self.gate_anchor;
    let ok = |cut: &Timestamp, min_gap: Duration, first_gate: bool| {
      if shot_start.is_some_and(|s| cut.cmp_semantic(&s).is_le()) {
        return false;
      }
      if let Some(l) = last_confirmed
        && !(cut.cmp_semantic(&l).is_gt() && cut.duration_since(&l).is_some_and(|d| d >= min_gap))
      {
        return false;
      }
      // With the lanes' internal gates zeroed, the machine also owns
      // the initial_cut(false) warm-up contract against the live
      // visual anchor (stream start or the latest dimension reset).
      if first_gate
        && let Some(anchor) = gate_anchor
        && cut.duration_since(&anchor).is_none_or(|d| d < min_gap)
      {
        return false;
      }
      true
    };

    // Terminal fade lane — cuts directly.
    let floor = self.options.threshold.threshold();
    let fade = match &mut self.threshold {
      Some(det) => det.process_luma(frames.luma()).map(|cut| {
        let mean = det.last_avg().unwrap_or_default();
        (cut, Provenance::Threshold(ThresholdScore { mean, floor }))
      }),
      None => None,
    };

    // The histogram / phash lanes are stateful: feed them every frame,
    // boundary or not. A firing lane passes the machine-side gate for
    // ITS configured spacing / warm-up (the lane itself is
    // policy-free), then cuts directly.
    let mut reports: Vec<(Timestamp, Provenance)> = Vec::new();
    if let Some(det) = &mut self.luma_hist
      && let Some(cut) = det.process(frames.luma())
    {
      let correlation = det.last_hist_diff().unwrap_or_default();
      if ok(
        &cut,
        self.options.luma_histogram.min_duration(),
        !self.options.luma_histogram.initial_cut(),
      ) {
        let plane = HistogramPlane::Luma;
        reports.push((
          cut,
          Provenance::Histogram(HistogramScore { plane, correlation }),
        ));
      }
    }
    if let Some(det) = &mut self.hue_hist {
      let hue = plane_as_luma(frames.hsv().hue(), frames.hsv(), ts);
      if let Some(cut) = det.process(hue) {
        let correlation = det.last_hist_diff().unwrap_or_default();
        if ok(
          &cut,
          self.options.hue_histogram.min_duration(),
          !self.options.hue_histogram.initial_cut(),
        ) {
          let plane = HistogramPlane::Hue;
          reports.push((
            cut,
            Provenance::Histogram(HistogramScore { plane, correlation }),
          ));
        }
      }
    }
    if let Some(det) = &mut self.sat_hist {
      let sat = plane_as_luma(frames.hsv().saturation(), frames.hsv(), ts);
      if let Some(cut) = det.process(sat) {
        let correlation = det.last_hist_diff().unwrap_or_default();
        if ok(
          &cut,
          self.options.sat_histogram.min_duration(),
          !self.options.sat_histogram.initial_cut(),
        ) {
          let plane = HistogramPlane::Saturation;
          reports.push((
            cut,
            Provenance::Histogram(HistogramScore { plane, correlation }),
          ));
        }
      }
    }
    if let Some(det) = &mut self.phash
      && let Some(cut) = det.process(frames.luma())
    {
      let distance = det.last_distance().unwrap_or_default();
      if ok(
        &cut,
        self.options.phash.min_duration(),
        !self.options.phash.initial_cut(),
      ) {
        reports.push((cut, Provenance::Phash(PhashScore { distance })));
      }
    }

    // The content / adaptive lanes run as plain detectors, options
    // verbatim.
    if let Some(det) = &mut self.cont_content
      && let Some(cut) = det.process_hsv(frames.hsv())
      && ok(
        &cut,
        self.options.content.min_duration(),
        !self.options.content.initial_cut(),
      )
    {
      let provenance = Provenance::Content(ContentScore {
        score: det.last_score().unwrap_or_default(),
        threshold: self.options.content.threshold(),
        components: det
          .last_components()
          .unwrap_or_else(|| content::Components::new(0.0, 0.0, 0.0, 0.0)),
      });
      reports.push((cut, provenance));
    }
    if let Some(det) = &mut self.cont_adaptive
      && let Some(cut) = det.process_hsv(frames.hsv())
      && ok(
        &cut,
        self.options.adaptive.min_duration(),
        !self.options.adaptive.initial_cut(),
      )
    {
      let provenance = Provenance::Adaptive(AdaptiveScore {
        ratio: det.last_adaptive_ratio().unwrap_or_default(),
        // The TARGET score: adaptive emits window_width frames
        // behind, so the trailing frame's score may be quiet at the
        // moment the lagged cut is reported.
        score: det.last_target_score().unwrap_or_default(),
      });
      reports.push((cut, provenance));
    }
    // The fade keeps its old tie precedence (front of the stable
    // sort); deferred survivors from earlier pushes contend last,
    // re-gated against the advanced shot state — an entry failing
    // its lane's gate lost the spacing race and drops for good.
    if let Some(f) = fade.filter(|(cut, _)| {
      ok(
        cut,
        self.options.threshold.min_duration(),
        !self.options.threshold.initial_cut(),
      )
    }) {
      reports.insert(0, f);
    }
    for report in mem::take(&mut self.deferred_direct) {
      let Some(report) = self.revalidate_deferred(report) else {
        continue;
      };
      let min_gap = self.lane_gate(&report.1);
      if ok(&report.0, min_gap, false) {
        reports.push(report);
      }
    }
    // The earliest admissible report closes the shot; later reports
    // from the same push are DEFERRED, not discarded — their lanes
    // consumed the transition evidence, so dropping them would erase
    // a tightly-spaced later boundary outright. Strictly-later only:
    // an equal-timestamp report is the same boundary seen by another
    // lane.
    // The forced-split cap contends last — it fires AT the current
    // frame, so any direct report this push wins or ties ahead of it.
    if let Some(split) = self.forced_split(ts) {
      reports.push(split);
    }
    reports.sort_by(|a, b| a.0.cmp_semantic(&b.0));
    let mut drained = reports.into_iter();
    let winner = drained.next();
    if let Some((cut, _)) = &winner {
      self
        .deferred_direct
        .extend(drained.filter(|(later, _)| later.cmp_semantic(cut).is_gt()));
      return winner;
    }
    None
  }

  /// The forced-split candidate: fires AT the current frame once the
  /// open shot's span reaches the cap (>=), so the closed range never
  /// exceeds the configured maximum at exact-cap frame timestamps. It
  /// contends with the detector boundaries at the lowest tie
  /// precedence — any report this push is at or before the current
  /// frame, so a real detection always beats the structural split.
  fn forced_split(&self, ts: Timestamp) -> Option<(Timestamp, Provenance)> {
    if let (Some(cap), Some(start)) = (self.options.max_scene_duration, self.shot_start)
      && let Some(span) = ts.duration_since(&start)
      && span >= cap
    {
      return Some((ts, Provenance::MaxSpan(SpanScore { span, cap })));
    }
    None
  }

  /// Revalidates a deferred report against the CURRENT shot. A forced
  /// split measured a PREVIOUS shot's span: once an earlier cut closes
  /// first, the split must hold against the new shot start —
  /// recomputed, or dropped until the cap genuinely trips again.
  /// Detector reports pass through: their timestamps are
  /// shot-independent observations.
  fn revalidate_deferred(
    &self,
    report: (Timestamp, Provenance),
  ) -> Option<(Timestamp, Provenance)> {
    match report.1 {
      Provenance::MaxSpan(_) => {
        let cap = self.options.max_scene_duration?;
        let start = self.shot_start?;
        let span = report.0.duration_since(&start)?;
        (span >= cap).then_some((report.0, Provenance::MaxSpan(SpanScore { span, cap })))
      }
      _ => Some(report),
    }
  }

  /// The configured spacing for the lane that produced a deferred
  /// report. Re-contention re-checks ordering and spacing ONLY: the
  /// warm-up gate was already passed at observation time, and the
  /// anchor it measures against re-seeds on dimension resets — a
  /// pre-reset observation must not be re-judged by a post-reset
  /// anchor.
  fn lane_gate(&self, provenance: &Provenance) -> Duration {
    match provenance {
      Provenance::Threshold(_) => self.options.threshold.min_duration(),
      Provenance::Histogram(s) => match s.plane() {
        HistogramPlane::Luma => self.options.luma_histogram.min_duration(),
        HistogramPlane::Hue => self.options.hue_histogram.min_duration(),
        HistogramPlane::Saturation => self.options.sat_histogram.min_duration(),
      },
      Provenance::Phash(_) => self.options.phash.min_duration(),
      Provenance::Content(_) => self.options.content.min_duration(),
      Provenance::Adaptive(_) => self.options.adaptive.min_duration(),
      // MaxSpan (the only structural provenance that defers) carries
      // no lane spacing; it is revalidated separately.
      _ => Duration::ZERO,
    }
  }

  /// Stale-echo and debounce guard: a confirmed timestamp at or before
  /// the open shot's start (or the last confirmed cut) is not a new
  /// boundary.
  fn admissible(
    &mut self,
    boundary: (Timestamp, Provenance),
    min_gap: Duration,
    first_gate: bool,
  ) -> Option<(Timestamp, Provenance)> {
    let (cut, _) = &boundary;
    if let Some(start) = self.shot_start
      && cut.cmp_semantic(&start).is_le()
    {
      return None;
    }
    // The lanes' internal debounce is zeroed for direct cuts, so the
    // machine owns both the inter-cut spacing and (for lanes
    // configured initial_cut(false)) the warm-up against the live
    // visual anchor.
    if let Some(last) = self.last_confirmed
      && !(cut.cmp_semantic(&last).is_gt()
        && cut.duration_since(&last).is_some_and(|d| d >= min_gap))
    {
      return None;
    }
    if first_gate
      && let Some(anchor) = self.gate_anchor
      && cut.duration_since(&anchor).is_none_or(|d| d < min_gap)
    {
      return None;
    }
    Some(boundary)
  }

  // --- shot lifecycle ---------------------------------------------------------

  /// Closes the shot at `cut` (exclusive) and re-anchors the keyframe
  /// window. The keyframes for `[shot_start, cut)` were already emitted
  /// by `finalize_window(cut)` (called just before this in `push` /
  /// `finalize`), so this only emits the scene and advances state.
  fn close_shot_at(&mut self, cut: Timestamp, provenance: Provenance) {
    let Some(start) = self.shot_start else {
      return;
    };
    // A shot that collapses to an instant under the canonical timebase
    // emits no scene (its keyframes parent to the neighboring range by
    // timestamp).
    let tb = self.stream_timebase.unwrap_or_else(|| cut.timebase());
    let range = shot_range(start, cut, tb);
    if let Some(range) = range {
      self.enqueue(Event::Scene(SceneEvent { range, provenance }));
    }
    // Open the next shot at the boundary and restart the keyframe
    // window there: frames at or after the cut stayed buffered in the
    // selector and belong to the new scene's first window.
    self.shot_start = Some(cut);
    self.last_confirmed = Some(cut);
    self.last_finalize = Some(cut);
  }

  /// Appends to the output backlog, shedding the oldest queued
  /// KEYFRAME once the backlog passes [`MAX_PENDING_OUTPUTS`]. Scenes
  /// are never shed: they enter at most once per push, so they cannot
  /// outgrow the one-per-push drain on their own.
  fn enqueue(&mut self, output: Event) {
    self.pending.push_back(output);
    if self.pending.len() > MAX_PENDING_OUTPUTS
      && let Some(idx) = self.pending.iter().position(|o| o.is_keyframe())
    {
      self.pending.remove(idx);
    }
  }

  // --- keyframe path ----------------------------------------------------------

  /// Computes one frame's [`FrameMetrics`] for the keyframe selector,
  /// running the staged extractor cascade: the cheap stage-3 metrics
  /// (luma mean / variance, saturation variance, clipping) always run;
  /// only frames passing the cheap [`hard_gate`] pay for the enabled
  /// expensive stage-4 metrics (sharpness, noise, motion blur,
  /// colorfulness). A cheap-gate failure leaves the expensive fields at
  /// zero — the selector then surfaces such a frame as a
  /// [`Provenance::Fallback`] pick. Strict-vs-fallback and the
  /// composite ranking are decided inside [`select::Detector`].
  fn frame_metrics(&mut self, frames: &Frames<'_>) -> FrameMetrics {
    let stats = self.luma_stats.observe_luma(frames.luma());
    let cheap = FrameMetrics::new()
      .with_brightness(stats.mean())
      .with_luma_variance(stats.variance())
      .with_saturation_variance(self.saturation.observe_hsv(frames.hsv()))
      .with_clipping(self.clipping.observe_rgb(frames.rgb()));
    if hard_gate(&cheap, &self.options.select) {
      return cheap;
    }
    // Stage 4: the enabled expensive metrics (a disabled extractor's
    // metric stays zero and its dedicated gate is skipped downstream).
    let extractors = self.options.extractors;
    let mut metrics = cheap;
    if extractors.contains(Extractors::SHARPNESS) {
      metrics = metrics.with_sharpness(self.sharpness.observe_luma(frames.luma()));
    }
    if extractors.contains(Extractors::NOISE) {
      metrics = metrics.with_noise(self.noise.observe_luma(frames.luma()));
    }
    if extractors.contains(Extractors::MOTION_BLUR) {
      metrics = metrics.with_motion_blur(self.motion_blur.observe_luma(frames.luma()));
    }
    if extractors.contains(Extractors::COLORFULNESS) {
      metrics = metrics.with_colorfulness(self.colorfulness.observe_rgb(frames.rgb()));
    }
    metrics
  }

  // --- resets -------------------------------------------------------------------

  /// Resets the visual stream state (the terminal fade lane and every
  /// per-frame lane); shot bookkeeping survives. The keyframe selector
  /// is deliberately untouched — a resolution switch does not reset
  /// keyframe selection: frames keep flowing into the same scene, and
  /// the in-flight window is drained by the normal `finalize_window`
  /// path in `push`, not here.
  fn reset_visual(&mut self) {
    // Deferred boundaries SURVIVE: they are timeline observations
    // already past their lane gates (like last_confirmed), not visual
    // state — a reset must not erase a deferred cut or the cap
    // split's original timestamp.
    if let Some(d) = &mut self.threshold {
      d.clear();
    }
    if let Some(d) = &mut self.luma_hist {
      d.clear();
    }
    if let Some(d) = &mut self.hue_hist {
      d.clear();
    }
    if let Some(d) = &mut self.sat_hist {
      d.clear();
    }
    if let Some(d) = &mut self.phash {
      d.clear();
    }
    if let Some(d) = &mut self.cont_content {
      d.clear();
    }
    if let Some(d) = &mut self.cont_adaptive {
      d.clear();
    }
  }

  /// Full refresh for the next stream.
  fn reset_all(&mut self) {
    self.reset_visual();
    self.dims = None;
    self.shot_start = None;
    self.stream_timebase = None;
    self.gate_anchor = None;
    self.last_ts = None;
    self.last_confirmed = None;
    self.selector.clear();
    self.last_finalize = None;
    self.recent_ts.clear();
    self.deferred_direct.clear();
    self.pending.clear();
  }
}

/// A single HSV plane viewed as a [`LumaFrame`] so a [`histogram`]
/// lane can watch its distribution (the chroma guards).
fn plane_as_luma<'a>(plane: &'a [u8], hsv: HsvFrame<'a>, ts: Timestamp) -> LumaFrame<'a> {
  LumaFrame::new(plane, hsv.width(), hsv.height(), hsv.stride(), ts)
}

/// The half-open shot range `[start, end)` in the stream's canonical
/// timebase; `None` when the pair is reversed or instant (a
/// real-but-sub-tick shot under a canonical timebase coarser than its
/// boundaries collapses and is skipped). Both endpoints rescale with
/// the same CEILING, which buys two properties at once: consecutive
/// ranges stay exactly adjacent (a partition), and the emitted ranges
/// tile the whole stream, so every delivered keyframe timestamp falls
/// inside some emitted range even when its own sub-tick scene was
/// skipped (it then parents to the neighboring range — the documented
/// degradation under hostile mixed-timebase input, instead of being
/// orphaned).
fn shot_range(start: Timestamp, end: Timestamp, tb: Timebase) -> Option<TimeRange> {
  let start_pts = rescale_ceil(start, tb);
  let end_pts = rescale_ceil(end, tb);
  TimeRange::try_new(start_pts, end_pts, tb).filter(|r| !r.is_instant())
}

/// Rescales `ts` into `tb`, rounding toward positive infinity — the
/// truncating rescale would shrink range ends below contained content.
/// i128 intermediates: `|pts| * num * den` peaks below 2^127. `None`
/// when the result does not fit an `i64` pts; push admission rejects
/// such frames up front, so range arithmetic never observes it.
fn rescale_ceil_checked(ts: Timestamp, tb: Timebase) -> Option<i64> {
  let num = i128::from(ts.pts()) * i128::from(ts.timebase().num()) * i128::from(tb.den().get());
  let den = i128::from(ts.timebase().den().get()) * i128::from(tb.num());
  let q = num.div_euclid(den);
  let ceil = if num.rem_euclid(den) != 0 { q + 1 } else { q };
  i64::try_from(ceil).ok()
}

/// The total form of [`rescale_ceil_checked`]: clamps at the
/// representable limits (unreachable after push admission; kept total
/// so range arithmetic needs no unwraps).
fn rescale_ceil(ts: Timestamp, tb: Timebase) -> i64 {
  rescale_ceil_checked(ts, tb).unwrap_or(if ts.pts() < 0 { i64::MIN } else { i64::MAX })
}

#[cfg(test)]
mod tests;
