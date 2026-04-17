#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use libm::{
  ceilf as ceil_32, cosf as cos_32, floorf as floor_32, round as round_64, roundf as round_32,
  sqrt as sqrt_64, sqrtf as sqrt_32,
};

/// Histogram-based scene detector using YUV luma correlation.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod histogram;

/// Perceptual hash-based scene detector using the DCT-based pHash algorithm.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod phash;

/// Intensity-threshold scene detector for fade-in / fade-out transitions.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod threshold;

/// Content-change scene detector using HSV-space per-frame deltas and
/// optional Canny edge comparison.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod content;

/// Rolling-average / adaptive scene detector built on top of the content
/// detector's scores. Reduces false positives on fast camera motion.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub mod adaptive;

/// Frame types for scene detection.
pub mod frame;

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn sqrt_64(val: f64) -> f64 {
  val.sqrt()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn sqrt_32(val: f32) -> f32 {
  val.sqrt()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn cos_32(val: f32) -> f32 {
  val.cos()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn floor_32(val: f32) -> f32 {
  val.floor()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn ceil_32(val: f32) -> f32 {
  val.ceil()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn round_64(val: f64) -> f64 {
  val.round()
}

#[cfg(feature = "std")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn round_32(val: f32) -> f32 {
  val.round()
}
