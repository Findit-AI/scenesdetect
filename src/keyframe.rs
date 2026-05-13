//! Keyframe metric detectors and the selection state machine.
//!
//! This module is deliberately minimal: each detector is a
//! single-concern, pure-algo state machine that scores one aspect of a
//! frame's quality. Composition — buffering scores, reacting to
//! shot-boundary events from the cut detectors — lives in the caller.
//!
//! Public modules:
//!
//! - [`preprocess`] — [`Downscaler`](preprocess::Downscaler),
//!   [`LumaConverter`](preprocess::LumaConverter),
//!   [`HsvConverter`](preprocess::HsvConverter): shared per-frame
//!   preprocessing utilities that own scratch buffers and hand out
//!   borrowed frames.
//! - [`luma`] — mean + variance of a luma plane.
//! - [`clipping`] — fraction of pixels whose brightest channel is
//!   crushed or blown.
//! - [`saturation`] — variance of the HSV saturation plane.
//! - [`sharpness`] — Tenengrad focus measure from a 3×3 Sobel on luma.
//! - [`score`] — the composite [`score::FrameScore`] type assembled
//!   from the four metric detectors above.
//! - [`select`] — buffers [`score::FrameScore`]s and emits keyframe
//!   timestamps on shot-boundary events.

pub mod clipping;
pub mod luma;
pub mod metrics;
pub mod preprocess;
pub mod saturation;
pub mod score;
pub mod select;
pub mod sharpness;

// Crate-internal shared reduction kernels. Not a public module — both
// `luma` and `saturation` reach into it via `super::reduce::...`.
mod reduce;
