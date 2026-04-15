#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

/// Histogram-based scene detector using YUV luma correlation.
pub mod histogram;

/// Perceptual hash-based scene detector using the DCT-based pHash algorithm.
pub mod phash;

/// Intensity-threshold scene detector for fade-in / fade-out transitions.
pub mod threshold;

/// Frame types for scene detection.
pub mod frame;
