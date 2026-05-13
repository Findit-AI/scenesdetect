//! Composite per-frame quality metrics.
//!
//! [`FrameMetrics`] bundles the eight per-frame quality measurements
//! produced by the per-metric detectors in this module's siblings
//! ([`sharpness`](crate::keyframe::sharpness),
//! [`luma`](crate::keyframe::luma),
//! [`saturation`](crate::keyframe::saturation),
//! [`clipping`](crate::keyframe::clipping),
//! [`noise`](crate::keyframe::noise),
//! [`motion_blur`](crate::keyframe::motion_blur),
//! [`colorfulness`](crate::keyframe::colorfulness)) into the shape
//! consumed by [`select::Detector`](crate::keyframe::select::Detector).
//!
//! Fields are private. Read access goes through the field-name
//! getters (`sharpness()` etc.); construction goes through `new()`
//! followed by `with_*` consuming builders (`const fn`) or `set_*`
//! mutating setters returning `&mut Self` for chaining.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// All per-frame quality measurements consumed by
/// [`select::Detector`](crate::keyframe::select::Detector).
///
/// Built by the per-metric detectors in this module's siblings. See the
/// module docs for the production pattern; `Default` yields the all-zero
/// `FrameMetrics`, useful in tests.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameMetrics {
  sharpness: f32,
  brightness: f32,
  luma_variance: f32,
  saturation_variance: f32,
  clipping: f32,
  noise: f32,
  motion_blur: f32,
  colorfulness: f32,
}

impl Default for FrameMetrics {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl FrameMetrics {
  /// Creates an all-zero [`FrameMetrics`] (same value as
  /// [`FrameMetrics::default`], but usable in `const` contexts).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      sharpness: 0.0,
      brightness: 0.0,
      luma_variance: 0.0,
      saturation_variance: 0.0,
      clipping: 0.0,
      noise: 0.0,
      motion_blur: 0.0,
      colorfulness: 0.0,
    }
  }

  // ---- Getters (const fn) ------------------------------------------------

  /// Tenengrad sharpness score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn sharpness(&self) -> f32 {
    self.sharpness
  }
  /// Luma mean in 0-255 space (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn brightness(&self) -> f32 {
    self.brightness
  }
  /// Luma population variance (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_variance(&self) -> f32 {
    self.luma_variance
  }
  /// HSV saturation-plane population variance (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn saturation_variance(&self) -> f32 {
    self.saturation_variance
  }
  /// Fraction of pixels whose brightest channel is clipped, in `[0, 1]`
  /// (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clipping(&self) -> f32 {
    self.clipping
  }
  /// Immerkaer noise estimate σₙ in 0-255 space (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn noise(&self) -> f32 {
    self.noise
  }
  /// Gradient anisotropy in `[0, 1]` (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn motion_blur(&self) -> f32 {
    self.motion_blur
  }
  /// Hasler-Süßstrunk colorfulness score (read-only accessor).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn colorfulness(&self) -> f32 {
    self.colorfulness
  }

  // ---- Consuming builders (const fn) -------------------------------------

  /// Returns `self` with [`sharpness`](Self::sharpness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_sharpness(mut self, v: f32) -> Self {
    self.sharpness = v;
    self
  }
  /// Returns `self` with [`brightness`](Self::brightness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_brightness(mut self, v: f32) -> Self {
    self.brightness = v;
    self
  }
  /// Returns `self` with [`luma_variance`](Self::luma_variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_luma_variance(mut self, v: f32) -> Self {
    self.luma_variance = v;
    self
  }
  /// Returns `self` with [`saturation_variance`](Self::saturation_variance) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_saturation_variance(mut self, v: f32) -> Self {
    self.saturation_variance = v;
    self
  }
  /// Returns `self` with [`clipping`](Self::clipping) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_clipping(mut self, v: f32) -> Self {
    self.clipping = v;
    self
  }
  /// Returns `self` with [`noise`](Self::noise) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_noise(mut self, v: f32) -> Self {
    self.noise = v;
    self
  }
  /// Returns `self` with [`motion_blur`](Self::motion_blur) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_motion_blur(mut self, v: f32) -> Self {
    self.motion_blur = v;
    self
  }
  /// Returns `self` with [`colorfulness`](Self::colorfulness) set to `v`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_colorfulness(mut self, v: f32) -> Self {
    self.colorfulness = v;
    self
  }

  // ---- Mutating setters (chainable) --------------------------------------

  /// In-place setter for [`sharpness`](Self::sharpness). Returns `&mut Self` for chaining.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_sharpness(&mut self, v: f32) -> &mut Self {
    self.sharpness = v;
    self
  }
  /// In-place setter for [`brightness`](Self::brightness).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_brightness(&mut self, v: f32) -> &mut Self {
    self.brightness = v;
    self
  }
  /// In-place setter for [`luma_variance`](Self::luma_variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_variance(&mut self, v: f32) -> &mut Self {
    self.luma_variance = v;
    self
  }
  /// In-place setter for [`saturation_variance`](Self::saturation_variance).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_saturation_variance(&mut self, v: f32) -> &mut Self {
    self.saturation_variance = v;
    self
  }
  /// In-place setter for [`clipping`](Self::clipping).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_clipping(&mut self, v: f32) -> &mut Self {
    self.clipping = v;
    self
  }
  /// In-place setter for [`noise`](Self::noise).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_noise(&mut self, v: f32) -> &mut Self {
    self.noise = v;
    self
  }
  /// In-place setter for [`motion_blur`](Self::motion_blur).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_motion_blur(&mut self, v: f32) -> &mut Self {
    self.motion_blur = v;
    self
  }
  /// In-place setter for [`colorfulness`](Self::colorfulness).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_colorfulness(&mut self, v: f32) -> &mut Self {
    self.colorfulness = v;
    self
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn default_is_all_zero() {
    let m = FrameMetrics::default();
    assert_eq!(m.sharpness(), 0.0);
    assert_eq!(m.brightness(), 0.0);
    assert_eq!(m.luma_variance(), 0.0);
    assert_eq!(m.saturation_variance(), 0.0);
    assert_eq!(m.clipping(), 0.0);
    assert_eq!(m.noise(), 0.0);
    assert_eq!(m.motion_blur(), 0.0);
    assert_eq!(m.colorfulness(), 0.0);
  }

  #[test]
  fn new_matches_default() {
    assert_eq!(FrameMetrics::new(), FrameMetrics::default());
  }

  #[test]
  fn new_is_const_context_usable() {
    const M: FrameMetrics = FrameMetrics::new();
    assert_eq!(M, FrameMetrics::default());
  }

  #[test]
  fn with_builders_roundtrip_through_getters() {
    let m = FrameMetrics::new()
      .with_sharpness(1.0)
      .with_brightness(2.0)
      .with_luma_variance(3.0)
      .with_saturation_variance(4.0)
      .with_clipping(0.5)
      .with_noise(6.0)
      .with_motion_blur(0.7)
      .with_colorfulness(8.0);
    assert_eq!(m.sharpness(), 1.0);
    assert_eq!(m.brightness(), 2.0);
    assert_eq!(m.luma_variance(), 3.0);
    assert_eq!(m.saturation_variance(), 4.0);
    assert_eq!(m.clipping(), 0.5);
    assert_eq!(m.noise(), 6.0);
    assert_eq!(m.motion_blur(), 0.7);
    assert_eq!(m.colorfulness(), 8.0);
  }

  #[test]
  fn with_builder_is_const_context_usable() {
    const M: FrameMetrics = FrameMetrics::new().with_sharpness(42.0).with_noise(5.0);
    assert_eq!(M.sharpness(), 42.0);
    assert_eq!(M.noise(), 5.0);
    // Other fields untouched.
    assert_eq!(M.brightness(), 0.0);
  }

  #[test]
  fn set_methods_chain_and_mutate() {
    let mut m = FrameMetrics::new();
    m.set_sharpness(10.0)
      .set_brightness(20.0)
      .set_luma_variance(30.0)
      .set_saturation_variance(40.0)
      .set_clipping(0.25)
      .set_noise(50.0)
      .set_motion_blur(0.6)
      .set_colorfulness(70.0);
    assert_eq!(m.sharpness(), 10.0);
    assert_eq!(m.brightness(), 20.0);
    assert_eq!(m.luma_variance(), 30.0);
    assert_eq!(m.saturation_variance(), 40.0);
    assert_eq!(m.clipping(), 0.25);
    assert_eq!(m.noise(), 50.0);
    assert_eq!(m.motion_blur(), 0.6);
    assert_eq!(m.colorfulness(), 70.0);
  }
}
