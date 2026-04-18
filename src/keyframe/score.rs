//! Composite per-frame quality score.
//!
//! [`FrameScore`] is the plain-data type that bundles the five metrics
//! produced by the four per-metric detectors
//! ([`sharpness`](crate::keyframe::sharpness),
//! [`luma`](crate::keyframe::luma),
//! [`saturation`](crate::keyframe::saturation),
//! [`clipping`](crate::keyframe::clipping)) into the shape consumed by
//! [`select::Detector`](crate::keyframe::select::Detector).
//!
//! The type is in its own module so the "produced by several
//! detectors, consumed by one" relationship is visible in the path.
//! Selection-side gate thresholds that interpret these fields live in
//! [`select::Options`](crate::keyframe::select::Options).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// All five per-frame quality metrics needed by the selection gates,
/// bundled into a single plain-data struct.
///
/// Assemble this in the application by reading the outputs of the four
/// metric detectors ([`sharpness`](crate::keyframe::sharpness),
/// [`luma`](crate::keyframe::luma),
/// [`saturation`](crate::keyframe::saturation),
/// [`clipping`](crate::keyframe::clipping)) on the same frame.
/// `Default` yields the all-zero score, useful in tests.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameScore {
  /// Tenengrad sharpness score.
  pub sharpness: f32,
  /// Luma mean in 0-255 space.
  pub brightness: f32,
  /// Luma population variance.
  pub luma_variance: f32,
  /// HSV saturation-plane population variance.
  pub saturation_variance: f32,
  /// Fraction of pixels whose brightest channel is clipped, in `[0, 1]`.
  pub clipping: f32,
}
