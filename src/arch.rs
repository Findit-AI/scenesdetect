//! Platform-specific SIMD (plus a scalar fallback) for packed-BGR → planar
//! HSV conversion, mean-absolute-difference reduction, and the 3×3 Sobel
//! magnitude/direction kernel used by the content detector.
//!
//! Dispatch is a mix of compile-time `cfg` / `target_feature` selection
//! and, on `x86` / `x86_64` when `std` is enabled, runtime CPU-feature
//! detection. In particular:
//! - `aarch64` uses NEON selected at compile time because NEON is part of
//!   the base ISA.
//! - `wasm32` uses the wasm SIMD backend when `simd128` is enabled.
//! - `x86` / `x86_64` use runtime dispatch with `is_x86_feature_detected!`
//!   under `std` to pick AVX2, then SSSE3, then scalar; without `std`,
//!   compile-time `target_feature` gating selects the best available path.
//! - Other targets use the scalar fallback.
//!
//! Additional platforms can be added as sibling private modules exposing
//! the same internal entry points and wired into [`bgr_to_hsv_planes`]
//! through the appropriate `cfg` and/or dispatch branch.
//!
//! The module is crate-internal. Public-facing wrappers for the color-space
//! conversion live in `crate::frame::convert`; the mean-absolute-difference
//! and Sobel kernels remain internal and are consumed directly by the
//! content detector.

// Platform-specific modules, each exposing `pub(super) unsafe fn
// bgr_to_hsv_planes(...)`. Gated so each file is only compiled on matching
// targets — the source need not exist for other arches.

// Miri cannot interpret platform SIMD intrinsics — gate all SIMD modules
// on `not(miri)` so the dispatcher falls through to the scalar backend.
// Detector tests then still run under Miri (validating memory safety of
// the full pipeline) without hitting unsupported operations.

use crate::frame::ChannelOrder;

#[cfg(all(target_arch = "aarch64", not(miri)))]
mod neon;

// x86 SIMD modules are only reachable when either:
//   - `std` is enabled (runtime `is_x86_feature_detected!` dispatch), or
//   - the matching `target_feature` is set at compile time (no-std dispatch).
// Without either gate, the functions would compile but nothing calls them,
// producing dead-code warnings under `-D warnings`.
#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  any(feature = "std", target_feature = "ssse3"),
  not(miri),
))]
mod x86_ssse3;

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  any(feature = "std", target_feature = "sse4.1"),
  not(miri),
))]
mod x86_sse41;

#[cfg(all(
  any(target_arch = "x86", target_arch = "x86_64"),
  any(feature = "std", target_feature = "avx2"),
  not(miri),
))]
mod x86_avx2;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
mod wasm_simd128;

/// Converts a packed 24-bit BGR frame into three planar HSV buffers that
/// match OpenCV's `cv2.COLOR_BGR2HSV` semantics. Dispatches to the best
/// implementation available for the build target.
///
/// Dispatch matrix:
///
/// - `aarch64` → NEON (compile-time; NEON is in base ARMv8-A ISA).
/// - `wasm32` with `simd128` target feature → wasm SIMD.
/// - `x86` / `x86_64`:
///   - With `std`, runtime `is_x86_feature_detected!` picks AVX2 → SSSE3 → scalar.
///   - Without `std`, compile-time `target_feature` picks the best path.
/// - Everything else → scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)] // one branch per build config
#[allow(clippy::too_many_arguments)] // signature fixed by the 3-plane + dims + flag shape
pub(crate) fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  order: ChannelOrder,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::Scalar::bgr_to_hsv_planes(
      h_out, s_out, v_out, src, width, height, stride, order,
    );
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA — every aarch64 Rust
    // target has it. No runtime feature detection required.
    unsafe {
      neon::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
    }
    return;
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    unsafe {
      wasm_simd128::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
    }
    return;
  }

  // x86 runtime dispatch when std is available.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("avx2") {
      // SAFETY: runtime-checked above. AVX2 implies SSSE3 at the hardware
      // level; the callee is annotated with both target features.
      unsafe {
        x86_avx2::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
      }
      return;
    }
    if std::is_x86_feature_detected!("sse4.1") {
      // SAFETY: runtime-checked above.
      unsafe {
        x86_sse41::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
      }
      return;
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      unsafe {
        x86_ssse3::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
      }
      return;
    }
  }

  // x86 compile-time dispatch when std is off.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "avx2",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_avx2::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
    }
    return;
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_sse41::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
    }
    return;
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_ssse3::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
    }
    return;
  }

  // Fallback.
  scalar::Scalar::bgr_to_hsv_planes(h_out, s_out, v_out, src, width, height, stride, order);
}

/// Single-pixel scalar BGR → HSV, exposed for tests and for callers that
/// need to process stray pixels one at a time.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(dead_code)] // used only from tests in some build configurations
pub(crate) fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
  scalar::Scalar::bgr_to_hsv_pixel(b, g, r)
}

/// Sum of absolute per-element differences of two equal-length `u8` slices,
/// divided by `n`. Dispatches to the best SIMD backend or scalar based on
/// `use_simd`.
///
/// NEON uses `vabdq_u8` + `vpaddlq` accumulate. x86 uses `_mm_sad_epu8`
/// (a single-instruction SAD per 16 bytes). wasm uses widening subtract +
/// abs reduce. All produce the same numerical result as scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn mean_abs_diff(a: &[u8], b: &[u8], n: usize, use_simd: bool) -> f64 {
  debug_assert!(a.len() >= n && b.len() >= n);
  if n == 0 {
    return 0.0;
  }

  if use_simd {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
      // SAFETY: NEON is base ARMv8-A ISA.
      return unsafe { neon::mean_abs_diff(a, b, n) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      feature = "std",
      not(miri)
    ))]
    {
      if std::is_x86_feature_detected!("sse4.1") {
        // SAFETY: runtime-checked.
        return unsafe { x86_sse41::mean_abs_diff(a, b, n) };
      }
      if std::is_x86_feature_detected!("ssse3") {
        // SAFETY: runtime-checked.
        return unsafe { x86_ssse3::mean_abs_diff(a, b, n) };
      }
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "sse4.1",
      not(miri),
    ))]
    {
      return unsafe { x86_sse41::mean_abs_diff(a, b, n) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "ssse3",
      not(target_feature = "sse4.1"),
      not(miri),
    ))]
    {
      return unsafe { x86_ssse3::mean_abs_diff(a, b, n) };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
      return unsafe { wasm_simd128::mean_abs_diff(a, b, n) };
    }
  }

  scalar::Scalar::mean_abs_diff(a, b, n)
}

/// 3×3 Sobel: computes L1 magnitude (`|Gx| + |Gy|`) into `mag` and a
/// quantized gradient direction (0=horiz, 1=45°, 2=vert, 3=135°) into `dir`.
/// Border pixels stay zero. Dispatches to SIMD for the magnitude computation;
/// direction quantization is always scalar (branchy per pixel).
///
/// Frames narrower or shorter than 3 pixels have no interior, so the
/// dispatcher short-circuits by zero-filling `mag`/`dir` and returning.
/// This both matches the "no interior → all-zero output" contract every
/// scalar/SIMD backend already implements and prevents the SIMD inner
/// loops from running with `w - 1`/`h - 1` row bounds that can
/// underflow `usize` for degenerate dimensions.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn sobel(
  input: &[u8],
  mag: &mut [i32],
  dir: &mut [u8],
  w: usize,
  h: usize,
  use_simd: bool,
) {
  if w < 3 || h < 3 {
    mag.fill(0);
    dir.fill(0);
    return;
  }

  if use_simd {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
      return unsafe { neon::sobel(input, mag, dir, w, h) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      feature = "std",
      not(miri)
    ))]
    {
      if std::is_x86_feature_detected!("sse4.1") {
        return unsafe { x86_sse41::sobel(input, mag, dir, w, h) };
      }
      if std::is_x86_feature_detected!("ssse3") {
        return unsafe { x86_ssse3::sobel(input, mag, dir, w, h) };
      }
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "sse4.1",
      not(miri),
    ))]
    {
      return unsafe { x86_sse41::sobel(input, mag, dir, w, h) };
    }

    #[cfg(all(
      any(target_arch = "x86", target_arch = "x86_64"),
      not(feature = "std"),
      target_feature = "ssse3",
      not(target_feature = "sse4.1"),
      not(miri),
    ))]
    {
      return unsafe { x86_ssse3::sobel(input, mag, dir, w, h) };
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
    {
      return unsafe { wasm_simd128::sobel(input, mag, dir, w, h) };
    }
  }

  scalar::Scalar::sobel(input, mag, dir, w, h);
}

/// Converts a packed 24-bit BGR frame into a single-plane BT.601 luma
/// buffer. Uses the fixed-point approximation
/// `Y = (77·R + 150·G + 29·B) >> 8`; coefficients sum to 256 so the
/// output stays in `[0, 255]`.
///
/// Dispatch matrix matches [`bgr_to_hsv_planes`]: aarch64 NEON at
/// compile time; x86 runtime dispatch under `std` picks AVX2 → SSSE3 →
/// scalar; `no_std` x86 dispatches at compile time; wasm uses
/// `simd128` when enabled; everything else falls back to scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn bgr_to_luma(
  out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  order: ChannelOrder,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::Scalar::bgr_to_luma(out, src, width, height, stride, order);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    unsafe {
      neon::bgr_to_luma(out, src, width, height, stride, order);
    }
    return;
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    unsafe {
      wasm_simd128::bgr_to_luma(out, src, width, height, stride, order);
    }
    return;
  }

  // x86 runtime dispatch under std. For bgr_to_luma the reduction is
  // memory-bandwidth-bound, so the SSSE3 (128-bit) path is already
  // close to peak throughput; we don't ship a separate AVX2 kernel,
  // AVX2-capable hosts land here and run SSSE3.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      unsafe {
        x86_sse41::bgr_to_luma(out, src, width, height, stride, order);
      }
      return;
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      unsafe {
        x86_ssse3::bgr_to_luma(out, src, width, height, stride, order);
      }
      return;
    }
  }

  // x86 compile-time dispatch without std.
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(miri),
  ))]
  {
    unsafe {
      x86_sse41::bgr_to_luma(out, src, width, height, stride, order);
    }
    return;
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    unsafe {
      x86_ssse3::bgr_to_luma(out, src, width, height, stride, order);
    }
    return;
  }

  scalar::Scalar::bgr_to_luma(out, src, width, height, stride, order);
}

/// Counts pixels in a packed 24-bit BGR frame whose brightest channel
/// falls outside the `[CLIP_LOW, CLIP_HIGH] = [5, 250]` unclipped band
/// — i.e. `max(R, G, B) < 5` (crushed shadow) or `> 250` (blown
/// highlight). Byte-order-agnostic: same count for BGR and RGB order.
///
/// Returns the count of clipped pixels. Caller divides by
/// `width * height` for the ratio.
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn clipping_count(
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
  use_simd: bool,
) -> u64 {
  if !use_simd {
    return scalar::Scalar::clipping_count(src, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::clipping_count(src, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::clipping_count(src, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::clipping_count(src, width, height, stride) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::clipping_count(src, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(miri),
  ))]
  {
    return unsafe { x86_sse41::clipping_count(src, width, height, stride) };
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::clipping_count(src, width, height, stride) };
  }

  scalar::Scalar::clipping_count(src, width, height, stride)
}

/// Tenengrad sharpness on a luma plane: `mean(gx² + gy²)` over
/// interior pixels, where `gx` / `gy` come from a 3×3 Sobel. Honours
/// row stride. Frames narrower or shorter than 3 pixels have no
/// interior and yield `0.0`.
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn tenengrad(
  luma: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::tenengrad(luma, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::tenengrad(luma, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::tenengrad(luma, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::tenengrad(luma, width, height, stride) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::tenengrad(luma, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(miri),
  ))]
  {
    return unsafe { x86_sse41::tenengrad(luma, width, height, stride) };
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::tenengrad(luma, width, height, stride) };
  }

  scalar::Scalar::tenengrad(luma, width, height, stride)
}

/// Immerkaer (1996) fast noise variance estimator. Returns σₙ (the
/// estimated per-pixel additive-Gaussian noise standard deviation) in
/// 0-255 space.
///
/// Dispatch matrix mirrors [`tenengrad`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn noise(
  luma: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::noise(luma, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::noise(luma, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::noise(luma, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("avx2") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_avx2::noise(luma, width, height, stride) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_sse41::noise(luma, width, height, stride) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::noise(luma, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "avx2",
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_avx2::noise(luma, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_sse41::noise(luma, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::noise(luma, width, height, stride) };
  }

  scalar::Scalar::noise(luma, width, height, stride)
}

/// Magnitude-weighted gradient-direction concentration. Returns an
/// anisotropy score in `[0, 1]` — 0 = isotropic gradients, 1 = a single
/// dominant direction.
///
/// Inputs are the magnitude and quantized-direction planes produced by
/// [`sobel`]. `mag.len()` and `dir.len()` MUST each be at least
/// `width * height`; this is a public-API precondition because the
/// SIMD backends use raw pointer loads with no bounds checks. The
/// dispatcher panics on a too-short slice rather than relying on the
/// scalar path's bounds-checked indexing to catch it (UB would
/// otherwise be reachable from the safe-Rust
/// [`MotionBlur::observe_sobel`](crate::keyframe::motion_blur::MotionBlur::observe_sobel)
/// entry point).
///
/// Dispatch matrix mirrors [`tenengrad`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn gradient_anisotropy(
  mag: &[i32],
  dir: &[u8],
  width: usize,
  height: usize,
  use_simd: bool,
) -> f32 {
  // Validate slice lengths BEFORE any SIMD dispatch. The SIMD
  // backends read 4 i32 mag values (SSSE3) or 8 i32 mag values +
  // 8 dir bytes (NEON, wasm) per chunk via raw pointer loads;
  // an undersized slice would be unchecked OOB reads / UB. The
  // scalar path would panic via bounds-checked indexing on the
  // same inputs — fail uniformly and explicitly here.
  let n = width
    .checked_mul(height)
    .expect("gradient_anisotropy: width*height overflows usize");
  assert!(
    mag.len() >= n,
    "gradient_anisotropy: mag.len()={} but width*height={n}",
    mag.len()
  );
  assert!(
    dir.len() >= n,
    "gradient_anisotropy: dir.len()={} but width*height={n}",
    dir.len()
  );

  if !use_simd {
    return scalar::Scalar::gradient_anisotropy(mag, dir, width, height);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    return unsafe { neon::gradient_anisotropy(mag, dir, width, height) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    return unsafe { wasm_simd128::gradient_anisotropy(mag, dir, width, height) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("avx2") {
      return unsafe { x86_avx2::gradient_anisotropy(mag, dir, width, height) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::gradient_anisotropy(mag, dir, width, height) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      return unsafe { x86_ssse3::gradient_anisotropy(mag, dir, width, height) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "avx2",
    not(miri),
  ))]
  {
    return unsafe { x86_avx2::gradient_anisotropy(mag, dir, width, height) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    return unsafe { x86_sse41::gradient_anisotropy(mag, dir, width, height) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    return unsafe { x86_ssse3::gradient_anisotropy(mag, dir, width, height) };
  }

  scalar::Scalar::gradient_anisotropy(mag, dir, width, height)
}

/// Hasler-Süßstrunk colourfulness metric on packed 24-bit BGR.
/// See the scalar kernel for the formula.
///
/// Dispatch matrix mirrors [`tenengrad`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn colorfulness(
  bgr: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  order: ChannelOrder,
  use_simd: bool,
) -> f32 {
  if !use_simd {
    return scalar::Scalar::colorfulness(bgr, width, height, stride, order);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    return unsafe { neon::colorfulness(bgr, width, height, stride, order) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    return unsafe { wasm_simd128::colorfulness(bgr, width, height, stride, order) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("avx2") {
      return unsafe { x86_avx2::colorfulness(bgr, width, height, stride, order) };
    }
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::colorfulness(bgr, width, height, stride, order) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      return unsafe { x86_ssse3::colorfulness(bgr, width, height, stride, order) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "avx2",
    not(miri),
  ))]
  {
    return unsafe { x86_avx2::colorfulness(bgr, width, height, stride, order) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    return unsafe { x86_sse41::colorfulness(bgr, width, height, stride, order) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(target_feature = "avx2"),
    not(miri),
  ))]
  {
    return unsafe { x86_ssse3::colorfulness(bgr, width, height, stride, order) };
  }

  scalar::Scalar::colorfulness(bgr, width, height, stride, order)
}

/// Population mean and variance of a single-plane `u8` image. Honours
/// row stride. Uses a single-pass `E[X²] - E[X]²` formulation —
/// numerically safe for `u8` inputs evaluated in `f64` (variance tops
/// out around 65025, well within f64 precision).
///
/// Dispatch matrix mirrors [`bgr_to_luma`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(unreachable_code)]
pub(crate) fn plane_mean_variance(
  plane: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> (f32, f32) {
  if !use_simd {
    return scalar::Scalar::plane_mean_variance(plane, width, height, stride);
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  {
    // SAFETY: NEON is part of the base ARMv8-A ISA.
    return unsafe { neon::plane_mean_variance(plane, width, height, stride) };
  }

  #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
  {
    // SAFETY: simd128 target feature enabled at compile time.
    return unsafe { wasm_simd128::plane_mean_variance(plane, width, height, stride) };
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "std",
    not(miri)
  ))]
  {
    if std::is_x86_feature_detected!("sse4.1") {
      return unsafe { x86_sse41::plane_mean_variance(plane, width, height, stride) };
    }
    if std::is_x86_feature_detected!("ssse3") {
      // SAFETY: runtime-checked above.
      return unsafe { x86_ssse3::plane_mean_variance(plane, width, height, stride) };
    }
  }

  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "sse4.1",
    not(miri),
  ))]
  {
    return unsafe { x86_sse41::plane_mean_variance(plane, width, height, stride) };
  }
  #[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "std"),
    target_feature = "ssse3",
    not(target_feature = "sse4.1"),
    not(miri),
  ))]
  {
    // SAFETY: target feature enabled at compile time.
    return unsafe { x86_ssse3::plane_mean_variance(plane, width, height, stride) };
  }

  scalar::Scalar::plane_mean_variance(plane, width, height, stride)
}

// -----------------------------------------------------------------------------
// Scalar implementation — used as the fallback on non-aarch64 targets and
// as the reference for the single-pixel helper everywhere.
//
// Common (non-SIMD) code is grouped under a ZST with `impl` methods; only the
// platform-specific SIMD backends use free functions (which is idiomatic for
// intrinsic-heavy code where each function carries a `target_feature`
// attribute).
// -----------------------------------------------------------------------------

mod scalar {
  use super::ChannelOrder;
  use crate::round_32;

  /// Zero-sized namespace for the scalar BGR→HSV kernels.
  pub(super) struct Scalar;

  impl Scalar {
    /// Whole-plane scalar BGR→HSV. Used as the fallback on targets without
    /// a SIMD backend.
    // On aarch64 the planar function is unused (NEON wins); keep it around
    // as a correctness reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bgr_to_hsv_planes(
      h_out: &mut [u8],
      s_out: &mut [u8],
      v_out: &mut [u8],
      src: &[u8],
      width: u32,
      height: u32,
      stride: u32,
      order: ChannelOrder,
    ) {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      let (b_off, r_off) = match order {
        ChannelOrder::Bgr => (0, 2),
        ChannelOrder::Rgb => (2, 0),
      };
      for y in 0..h {
        let row = &src[y * s..y * s + w * 3];
        let dst_off = y * w;
        for x in 0..w {
          let b = row[x * 3 + b_off] as f32;
          let g = row[x * 3 + 1] as f32;
          let r = row[x * 3 + r_off] as f32;
          let (hue, sat, val) = Self::bgr_to_hsv_pixel(b, g, r);
          h_out[dst_off + x] = hue;
          s_out[dst_off + x] = sat;
          v_out[dst_off + x] = val;
        }
      }
    }

    /// Scalar BGR→HSV for a single pixel. Inputs are floats (typically from
    /// `u8 as f32`); outputs are clamped/rounded u8 in OpenCV's 8-bit
    /// encoding (H in [0, 179], S and V in [0, 255]).
    #[inline]
    pub(super) fn bgr_to_hsv_pixel(b: f32, g: f32, r: f32) -> (u8, u8, u8) {
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
      let h8 = round_32(hue * 0.5).clamp(0.0, 179.0) as u8;
      (
        h8,
        round_32(s).clamp(0.0, 255.0) as u8,
        round_32(v).clamp(0.0, 255.0) as u8,
      )
    }

    /// Scalar 3×3 Sobel: magnitude + direction.
    pub(super) fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], w: usize, h: usize) {
      mag.fill(0);
      dir.fill(0);
      for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
          let i = |yy: usize, xx: usize| input[yy * w + xx] as i32;
          let gx = -i(y - 1, x - 1) - 2 * i(y, x - 1) - i(y + 1, x - 1)
            + i(y - 1, x + 1)
            + 2 * i(y, x + 1)
            + i(y + 1, x + 1);
          let gy = -i(y - 1, x - 1) - 2 * i(y - 1, x) - i(y - 1, x + 1)
            + i(y + 1, x - 1)
            + 2 * i(y + 1, x)
            + i(y + 1, x + 1);
          let idx = y * w + x;
          mag[idx] = gx.abs() + gy.abs();
          let ax = gx.abs();
          let ay = gy.abs();
          dir[idx] = if ay * 1000 < ax * 414 {
            0
          } else if ay * 1000 > ax * 2414 {
            2
          } else if gx.signum() == gy.signum() {
            1
          } else {
            3
          };
        }
      }
    }

    /// Scalar mean absolute difference: `Σ|a[i] - b[i]| / n`.
    #[inline]
    pub(super) fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
      let mut sum: u64 = 0;
      for i in 0..n {
        let da = a[i] as i32 - b[i] as i32;
        sum += da.unsigned_abs() as u64;
      }
      sum as f64 / n as f64
    }

    /// Scalar BGR → BT.601 luma: `Y = (77·R + 150·G + 29·B) >> 8`.
    /// Honours row stride; writes tight-packed `width * height` bytes
    /// into `out`.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference for tests and the fallback elsewhere.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn bgr_to_luma(
      out: &mut [u8],
      src: &[u8],
      width: u32,
      height: u32,
      stride: u32,
      order: ChannelOrder,
    ) {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      let (b_off, r_off) = match order {
        ChannelOrder::Bgr => (0, 2),
        ChannelOrder::Rgb => (2, 0),
      };
      for y in 0..h {
        let row_off = y * s;
        let dst_off = y * w;
        let row = &src[row_off..row_off + w * 3];
        let dst = &mut out[dst_off..dst_off + w];
        for x in 0..w {
          let b = row[x * 3 + b_off] as u32;
          let g = row[x * 3 + 1] as u32;
          let r = row[x * 3 + r_off] as u32;
          dst[x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
      }
    }

    /// Scalar clipping-pixel count. A pixel is clipped iff
    /// `max(R, G, B) < 5` or `> 250`. Honours row stride.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn clipping_count(src: &[u8], width: u32, height: u32, stride: u32) -> u64 {
      let w = width as usize;
      let h = height as usize;
      let s = stride as usize;
      let mut count: u64 = 0;
      for y in 0..h {
        let row = &src[y * s..y * s + w * 3];
        for x in 0..w {
          let b = row[x * 3];
          let g = row[x * 3 + 1];
          let r = row[x * 3 + 2];
          let m = b.max(g).max(r);
          if !(5..=250).contains(&m) {
            count += 1;
          }
        }
      }
      count
    }

    /// Scalar single-pass `(mean, variance)` via u64 `sum_x` and
    /// `sum_x²`, finalized as `E[X²] - E[X]²` in f64. Honours row
    /// stride.
    // On aarch64 the NEON path wins; keep this as the correctness reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn plane_mean_variance(plane: &[u8], w: usize, h: usize, s: usize) -> (f32, f32) {
      let n = w.saturating_mul(h);
      if n == 0 {
        return (0.0, 0.0);
      }
      let mut sum: u64 = 0;
      let mut sum_sq: u64 = 0;
      for y in 0..h {
        let row = &plane[y * s..y * s + w];
        for &v in row {
          sum += v as u64;
          sum_sq += (v as u64) * (v as u64);
        }
      }
      let n_f = n as f64;
      let mean = (sum as f64) / n_f;
      let mean_sq = (sum_sq as f64) / n_f;
      let variance = (mean_sq - mean * mean).max(0.0);
      (mean as f32, variance as f32)
    }

    /// Scalar Tenengrad: `mean(gx² + gy²)` over interior pixels of a
    /// luma plane, where `gx`/`gy` are 3×3 Sobel gradients. Honours
    /// row stride.
    // On aarch64 the NEON path wins; keep this as the correctness
    // reference.
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(super) fn tenengrad(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let interior = (w - 2) * (h - 2);
      if interior == 0 {
        return 0.0;
      }

      let mut acc: i64 = 0;
      for y in 1..h - 1 {
        for x in 1..w - 1 {
          let p = |dy: isize, dx: isize| -> i32 {
            luma[((y as isize + dy) as usize) * s + ((x as isize + dx) as usize)] as i32
          };
          let tl = p(-1, -1);
          let t = p(-1, 0);
          let tr = p(-1, 1);
          let l = p(0, -1);
          let r = p(0, 1);
          let bl = p(1, -1);
          let b = p(1, 0);
          let br = p(1, 1);
          let gx = -tl - 2 * l - bl + tr + 2 * r + br;
          let gy = -tl - 2 * t - tr + bl + 2 * b + br;
          acc += (gx * gx + gy * gy) as i64;
        }
      }

      ((acc as f64) / (interior as f64)) as f32
    }

    /// Immerkaer (1996) fast noise variance estimator.
    /// Convolves the luma plane with the 3×3 Laplacian-of-difference
    /// mask `[[1,-2,1],[-2,4,-2],[1,-2,1]]`, sums the absolute values
    /// over interior pixels, then scales by `sqrt(π/2) / (6·N_inner)`
    /// to yield an estimate of the per-pixel additive-Gaussian noise
    /// standard deviation σₙ in 0-255 space. Honours row stride.
    pub(super) fn noise(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let interior = (w - 2) * (h - 2);
      if interior == 0 {
        return 0.0;
      }

      // Σ |I ⊛ N| over interior, where N is the Laplacian-of-difference
      // mask above. Per-pixel response fits in i32 (peak magnitude on
      // 8-bit input is 4·255 + 4·2·255 + 4·255 = 4·255 + 8·255 + 4·255 =
      // 16·255 = 4080); the i64 accumulator handles any realistic
      // interior count.
      let mut acc: i64 = 0;
      for y in 1..h - 1 {
        for x in 1..w - 1 {
          let p = |dy: isize, dx: isize| -> i32 {
            luma[((y as isize + dy) as usize) * s + ((x as isize + dx) as usize)] as i32
          };
          let tl = p(-1, -1);
          let t = p(-1, 0);
          let tr = p(-1, 1);
          let l = p(0, -1);
          let c = p(0, 0);
          let r = p(0, 1);
          let bl = p(1, -1);
          let b = p(1, 0);
          let br = p(1, 1);
          // N · I = 4c - 2(t+b+l+r) + (tl+tr+bl+br)
          let lap = 4 * c - 2 * (t + b + l + r) + (tl + tr + bl + br);
          acc += lap.unsigned_abs() as i64;
        }
      }

      // σₙ ≈ √(π/2) / 6 · (Σ|lap| / interior)
      // √(π/2) / 6 ≈ 0.2088987...
      const COEFF: f64 = 0.208_898_754_886_372_3;
      ((acc as f64) * COEFF / (interior as f64)) as f32
    }

    /// Magnitude-weighted gradient-direction concentration.
    /// Inputs are the magnitude and direction planes produced by
    /// [`super::super::sobel`] — `mag` is L1 magnitude (i32), `dir` is
    /// the quantized direction bin in `{0, 1, 2, 3}`. Border pixels
    /// (where Sobel leaves zeros) contribute nothing.
    ///
    /// Builds `hist[k] = Σ mag[p] where dir[p] == k`, then returns
    /// `max((max(hist) / total) - 0.25, 0) / 0.75`. The 0.25 baseline
    /// is the uniform-distribution expectation over the 4 bins; the
    /// output is in `[0, 1]` with 0 = perfectly uniform and 1 =
    /// entirely one direction. Frames with total magnitude 0 (uniform
    /// luma) return 0.
    pub(super) fn gradient_anisotropy(mag: &[i32], dir: &[u8], w: usize, h: usize) -> f32 {
      if w < 3 || h < 3 {
        return 0.0;
      }
      let mut hist = [0u64; 4];
      for y in 1..h - 1 {
        for x in 1..w - 1 {
          let idx = y * w + x;
          let m = mag[idx];
          if m <= 0 {
            continue;
          }
          let d = dir[idx] as usize & 0b11;
          hist[d] = hist[d].saturating_add(m as u64);
        }
      }
      let total: u64 = hist.iter().sum();
      if total == 0 {
        return 0.0;
      }
      let max_bin = *hist.iter().max().expect("4 bins") as f64;
      let total_f = total as f64;
      let frac = max_bin / total_f;
      // Normalise so [0.25, 1.0] maps to [0.0, 1.0]; clamp below.
      ((frac - 0.25).max(0.0) / 0.75) as f32
    }

    /// Hasler-Süßstrunk colourfulness metric on packed 24-bit BGR.
    /// Single pass, streaming moments on `rg = R - G` and
    /// `yb = 0.5(R + G) - B`. Returns
    /// `σ_rgyb + 0.3·μ_rgyb` where
    /// `σ_rgyb = √(σ²_rg + σ²_yb)` and `μ_rgyb = √(μ²_rg + μ²_yb)`.
    /// Empty inputs return 0.
    pub(super) fn colorfulness(
      bgr: &[u8],
      w: usize,
      h: usize,
      stride: usize,
      order: ChannelOrder,
    ) -> f32 {
      let n = w.saturating_mul(h);
      if n == 0 {
        return 0.0;
      }
      let n_f = n as f64;

      let (b_off, r_off) = match order {
        ChannelOrder::Bgr => (0, 2),
        ChannelOrder::Rgb => (2, 0),
      };

      // Welford-style streaming mean/M2 on rg and yb concurrently.
      let mut mean_rg: f64 = 0.0;
      let mut m2_rg: f64 = 0.0;
      let mut mean_yb: f64 = 0.0;
      let mut m2_yb: f64 = 0.0;
      let mut k: u64 = 0;

      for y in 0..h {
        let row = &bgr[y * stride..y * stride + w * 3];
        // BGR packed: row[3i + b_off] = B, row[3i+1] = G, row[3i + r_off] = R.
        for i in 0..w {
          let b = row[3 * i + b_off] as f64;
          let g = row[3 * i + 1] as f64;
          let r = row[3 * i + r_off] as f64;
          let rg = r - g;
          let yb = 0.5 * (r + g) - b;
          k += 1;
          let kf = k as f64;
          let d_rg = rg - mean_rg;
          mean_rg += d_rg / kf;
          m2_rg += d_rg * (rg - mean_rg);
          let d_yb = yb - mean_yb;
          mean_yb += d_yb / kf;
          m2_yb += d_yb * (yb - mean_yb);
        }
      }

      // Population variance (use k, not k-1) so an all-identical frame
      // gives σ = 0.
      //
      // `crate::sqrt_64` routes through the inherent `f64::sqrt` under
      // `std` and through `libm::sqrt` under `alloc`/no_std — the bare
      // `.sqrt()` method isn't available in the latter, so calling it
      // directly would break `cargo check --no-default-features
      // --features alloc`.
      let var_rg = (m2_rg / n_f).max(0.0);
      let var_yb = (m2_yb / n_f).max(0.0);
      let sigma_rgyb = crate::sqrt_64(var_rg + var_yb);
      let mu_rgyb = crate::sqrt_64(mean_rg * mean_rg + mean_yb * mean_yb);
      (sigma_rgyb + 0.3 * mu_rgyb) as f32
    }
  }
}

// ---------------------------------------------------------------------------
// Direct-call tests for platform SIMD backends. On x86 hosts, the runtime
// dispatcher picks AVX2 when available, leaving the SSSE3 `bgr_to_hsv_planes`
// path untested. These tests call each backend directly so coverage includes
// all compiled SIMD code regardless of which tier the host CPU supports.
// ---------------------------------------------------------------------------
// Miri: the scalar tests are fine, but the direct SIMD-call tests reference
// modules that are gated out under `cfg(miri)`. Gate the whole test module
// on `not(miri)` — Miri exercises the scalar paths through the detector-level
// tests in content.rs instead.
#[cfg(all(test, feature = "std", not(miri)))]
mod tests {
  use super::*;

  fn make_bgr(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    let mut rng = 0x9E3779B9u32;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  fn make_luma(w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h];
    let mut rng = 0xDEADBEEFu32;
    for v in buf.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    buf
  }

  // Exercises the scalar bgr_to_hsv_planes + mean_abs_diff + sobel.
  #[test]
  fn scalar_bgr_to_hsv_planes() {
    let (w, h) = (32, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    scalar::Scalar::bgr_to_hsv_planes(
      &mut ho,
      &mut so,
      &mut vo,
      &src,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Bgr,
    );
    assert!(vo.iter().any(|&v| v > 0));
  }

  // RGB byte order on the scalar path: the output must match
  // running BGR with R/B bytes manually swapped per pixel.
  #[test]
  fn scalar_rgb_to_hsv_planes_matches_swapped_bgr() {
    let (w, h) = (32, 16);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let n = w * h;
    let mut h_bgr = vec![0u8; n];
    let mut s_bgr = vec![0u8; n];
    let mut v_bgr = vec![0u8; n];
    scalar::Scalar::bgr_to_hsv_planes(
      &mut h_bgr,
      &mut s_bgr,
      &mut v_bgr,
      &bgr,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Bgr,
    );
    let mut h_rgb = vec![0u8; n];
    let mut s_rgb = vec![0u8; n];
    let mut v_rgb = vec![0u8; n];
    scalar::Scalar::bgr_to_hsv_planes(
      &mut h_rgb,
      &mut s_rgb,
      &mut v_rgb,
      &rgb,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Rgb,
    );
    assert_eq!(h_bgr, h_rgb);
    assert_eq!(s_bgr, s_rgb);
    assert_eq!(v_bgr, v_rgb);
  }

  #[test]
  fn scalar_mean_abs_diff_nonzero() {
    let a = make_luma(64, 1);
    let b = make_luma(64, 1);
    let d = scalar::Scalar::mean_abs_diff(&a, &b, 64);
    assert!(d >= 0.0);
  }

  #[test]
  fn scalar_sobel() {
    let (w, h) = (16, 16);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    scalar::Scalar::sobel(&src, &mut mag, &mut dir, w, h);
    assert!(mag.iter().any(|&m| m > 0));
  }

  #[test]
  fn sobel_dispatcher_short_circuits_on_zero_width() {
    // The dispatcher must guard against w < 3 or h < 3 *before*
    // routing to SIMD backends — the wasm path uses `w - 1` and `h - 1`
    // for its row bounds, which underflows `usize` on zero dimensions
    // and would drive aggressive unsafe loads.
    let mut mag = vec![0xFFi32; 16];
    let mut dir = vec![0xFFu8; 16];
    super::sobel(&[], &mut mag, &mut dir, 0, 4, false);
    // The dispatcher's contract: "no interior → all-zero output".
    assert!(mag.iter().all(|&m| m == 0));
    assert!(dir.iter().all(|&d| d == 0));
  }

  #[test]
  fn sobel_dispatcher_short_circuits_on_zero_height() {
    let mut mag = vec![0xFFi32; 16];
    let mut dir = vec![0xFFu8; 16];
    super::sobel(&[], &mut mag, &mut dir, 4, 0, false);
    assert!(mag.iter().all(|&m| m == 0));
    assert!(dir.iter().all(|&d| d == 0));
  }

  #[test]
  fn sobel_dispatcher_short_circuits_on_too_small() {
    // 2x2 also has no interior.
    let src = vec![0u8; 4];
    let mut mag = vec![0xFFi32; 4];
    let mut dir = vec![0xFFu8; 4];
    super::sobel(&src, &mut mag, &mut dir, 2, 2, false);
    assert!(mag.iter().all(|&m| m == 0));
    assert!(dir.iter().all(|&d| d == 0));
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn sobel_dispatcher_short_circuits_with_simd_enabled_on_zero_dims() {
    // Same guard must run even when SIMD is on.
    let mut mag = vec![0xFFi32; 16];
    let mut dir = vec![0xFFu8; 16];
    super::sobel(&[], &mut mag, &mut dir, 0, 8, true);
    assert!(mag.iter().all(|&m| m == 0));
    assert!(dir.iter().all(|&d| d == 0));
  }

  // x86: call SSSE3 bgr_to_hsv_planes directly (bypasses AVX2 dispatch).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_bgr_to_hsv_planes_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      x86_ssse3::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
    }
    // Sanity: V plane should have nonzero values for random input.
    assert!(vo.iter().any(|&v| v > 0));
  }

  // SSSE3 RGB path: must match the swapped-BGR scalar reference.
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_rgb_bgr_to_hsv_planes_matches_swapped() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (64, 16);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let n = w * h;
    let mut h_bgr = vec![0u8; n];
    let mut s_bgr = vec![0u8; n];
    let mut v_bgr = vec![0u8; n];
    let mut h_rgb = vec![0u8; n];
    let mut s_rgb = vec![0u8; n];
    let mut v_rgb = vec![0u8; n];
    unsafe {
      x86_ssse3::bgr_to_hsv_planes(
        &mut h_bgr,
        &mut s_bgr,
        &mut v_bgr,
        &bgr,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
      x86_ssse3::bgr_to_hsv_planes(
        &mut h_rgb,
        &mut s_rgb,
        &mut v_rgb,
        &rgb,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Rgb,
      );
    }
    assert_eq!(h_bgr, h_rgb);
    assert_eq!(s_bgr, s_rgb);
    assert_eq!(v_bgr, v_rgb);
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_mean_abs_diff_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let a = make_luma(128, 1);
    let b = make_luma(128, 1);
    let d = unsafe { x86_ssse3::mean_abs_diff(&a, &b, 128) };
    assert!(d >= 0.0);
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_sobel_direct() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (32, 32);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    unsafe { x86_ssse3::sobel(&src, &mut mag, &mut dir, w, h) };
    assert!(mag.iter().any(|&m| m > 0));
  }

  // x86: call AVX2 bgr_to_hsv_planes directly (exercises the AVX2 tail path too).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_bgr_to_hsv_planes_direct() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      x86_avx2::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
    }
    assert!(vo.iter().any(|&v| v > 0));
  }

  // AVX2 RGB path: must match the swapped-BGR run.
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_rgb_bgr_to_hsv_planes_matches_swapped() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    let (w, h) = (64, 16);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let n = w * h;
    let mut h_bgr = vec![0u8; n];
    let mut s_bgr = vec![0u8; n];
    let mut v_bgr = vec![0u8; n];
    let mut h_rgb = vec![0u8; n];
    let mut s_rgb = vec![0u8; n];
    let mut v_rgb = vec![0u8; n];
    unsafe {
      x86_avx2::bgr_to_hsv_planes(
        &mut h_bgr,
        &mut s_bgr,
        &mut v_bgr,
        &bgr,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
      x86_avx2::bgr_to_hsv_planes(
        &mut h_rgb,
        &mut s_rgb,
        &mut v_rgb,
        &rgb,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Rgb,
      );
    }
    assert_eq!(h_bgr, h_rgb);
    assert_eq!(s_bgr, s_rgb);
    assert_eq!(v_bgr, v_rgb);
  }

  // aarch64: call NEON bgr_to_hsv_planes directly.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_bgr_to_hsv_planes_direct() {
    let (w, h) = (64, 16);
    let src = make_bgr(w, h);
    let n = w * h;
    let mut ho = vec![0u8; n];
    let mut so = vec![0u8; n];
    let mut vo = vec![0u8; n];
    unsafe {
      neon::bgr_to_hsv_planes(
        &mut ho,
        &mut so,
        &mut vo,
        &src,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
    }
    assert!(vo.iter().any(|&v| v > 0));
  }

  // NEON RGB path: must match the swapped-BGR run.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_rgb_bgr_to_hsv_planes_matches_swapped() {
    let (w, h) = (64, 16);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let n = w * h;
    let mut h_bgr = vec![0u8; n];
    let mut s_bgr = vec![0u8; n];
    let mut v_bgr = vec![0u8; n];
    let mut h_rgb = vec![0u8; n];
    let mut s_rgb = vec![0u8; n];
    let mut v_rgb = vec![0u8; n];
    unsafe {
      neon::bgr_to_hsv_planes(
        &mut h_bgr,
        &mut s_bgr,
        &mut v_bgr,
        &bgr,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
      neon::bgr_to_hsv_planes(
        &mut h_rgb,
        &mut s_rgb,
        &mut v_rgb,
        &rgb,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Rgb,
      );
    }
    assert_eq!(h_bgr, h_rgb);
    assert_eq!(s_bgr, s_rgb);
    assert_eq!(v_bgr, v_rgb);
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_mean_abs_diff_direct() {
    let a = make_luma(128, 1);
    let b = make_luma(128, 1);
    let d = unsafe { neon::mean_abs_diff(&a, &b, 128) };
    assert!(d >= 0.0);
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_sobel_direct() {
    let (w, h) = (32, 32);
    let src = make_luma(w, h);
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    unsafe { neon::sobel(&src, &mut mag, &mut dir, w, h) };
    assert!(mag.iter().any(|&m| m > 0));
  }

  // ---- bgr_to_luma ----------------------------------------------------------

  #[test]
  fn scalar_bgr_to_luma_spot_values() {
    // Pure red / green / blue through the BT.601 fixed-point: the
    // per-pixel math is (77R + 150G + 29B) >> 8 where each channel is
    // exactly 255 in isolation.
    let red = [0u8, 0, 255];
    let green = [0u8, 255, 0];
    let blue = [255u8, 0, 0];
    let mut out = [0u8; 1];
    scalar::Scalar::bgr_to_luma(&mut out, &red, 1, 1, 3, ChannelOrder::Bgr);
    assert_eq!(out[0], 76);
    scalar::Scalar::bgr_to_luma(&mut out, &green, 1, 1, 3, ChannelOrder::Bgr);
    assert_eq!(out[0], 149);
    scalar::Scalar::bgr_to_luma(&mut out, &blue, 1, 1, 3, ChannelOrder::Bgr);
    assert_eq!(out[0], 28);
  }

  #[test]
  fn scalar_rgb_to_luma_matches_swapped_bgr() {
    // Same RGB-vs-swapped-BGR equivalence as for HSV.
    let (w, h) = (16usize, 8usize);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let mut out_bgr = vec![0u8; w * h];
    let mut out_rgb = vec![0u8; w * h];
    scalar::Scalar::bgr_to_luma(
      &mut out_bgr,
      &bgr,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Bgr,
    );
    scalar::Scalar::bgr_to_luma(
      &mut out_rgb,
      &rgb,
      w as u32,
      h as u32,
      (w * 3) as u32,
      ChannelOrder::Rgb,
    );
    assert_eq!(out_bgr, out_rgb);
  }

  // Scalar-equivalence test: the scalar reference and every SIMD
  // backend must produce byte-identical output. Uses a random-ish
  // fixed-seed pattern so the assertion is deterministic across
  // runs and CI targets.
  fn assert_bgr_to_luma_equiv<F: FnOnce(&mut [u8], &[u8], u32, u32, u32)>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    // Build a padded source where pixel bytes are random-ish and
    // padding bytes are a distinctive 0xAA (which would corrupt the
    // output if stride were ignored).
    let mut src = vec![0xAAu8; stride * h];
    let mut rng = 0x9E3779B9u32;
    for y in 0..h {
      for x in 0..w * 3 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        src[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let mut out_scalar = vec![0u8; w * h];
    scalar::Scalar::bgr_to_luma(
      &mut out_scalar,
      &src,
      w as u32,
      h as u32,
      stride as u32,
      ChannelOrder::Bgr,
    );
    let mut out_simd = vec![0u8; w * h];
    backend(&mut out_simd, &src, w as u32, h as u32, stride as u32);
    assert_eq!(out_simd, out_scalar, "SIMD backend disagrees with scalar");
  }

  // aarch64: call NEON bgr_to_luma directly, on various dims to
  // exercise both the 16-wide main loop and the scalar tail.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_bgr_to_luma_matches_scalar() {
    // Exactly 16 pixels wide — main loop only, no tail.
    assert_bgr_to_luma_equiv(16, 4, 16 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    // 17 pixels — one tail pixel per row.
    assert_bgr_to_luma_equiv(17, 4, 17 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    // Stride padding: 24 pixels per row, stride of 96 bytes.
    assert_bgr_to_luma_equiv(24, 5, 96, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    // Larger frame to catch any accumulator issues.
    assert_bgr_to_luma_equiv(257, 31, 257 * 3, |out, src, w, h, s| unsafe {
      neon::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
  }

  // NEON RGB byte-order path must match swapped-BGR.
  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_rgb_bgr_to_luma_matches_swapped() {
    let (w, h) = (24usize, 8usize);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let mut out_bgr = vec![0u8; w * h];
    let mut out_rgb = vec![0u8; w * h];
    unsafe {
      neon::bgr_to_luma(
        &mut out_bgr,
        &bgr,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
      neon::bgr_to_luma(
        &mut out_rgb,
        &rgb,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Rgb,
      );
    }
    assert_eq!(out_bgr, out_rgb);
  }

  // x86 SSSE3: call bgr_to_luma directly. Skips itself when the host
  // CPU doesn't support SSSE3 (the runtime check bypasses the SIMD
  // backend in that case, so we can't reach it anyway).
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_bgr_to_luma_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_bgr_to_luma_equiv(16, 4, 16 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    assert_bgr_to_luma_equiv(17, 4, 17 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    assert_bgr_to_luma_equiv(24, 5, 96, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
    assert_bgr_to_luma_equiv(257, 31, 257 * 3, |out, src, w, h, s| unsafe {
      x86_ssse3::bgr_to_luma(out, src, w, h, s, ChannelOrder::Bgr);
    });
  }

  // x86 SSSE3 RGB byte-order path must match swapped-BGR.
  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_rgb_bgr_to_luma_matches_swapped() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    let (w, h) = (24usize, 8usize);
    let bgr = make_bgr(w, h);
    let mut rgb = bgr.clone();
    for chunk in rgb.chunks_exact_mut(3) {
      chunk.swap(0, 2);
    }
    let mut out_bgr = vec![0u8; w * h];
    let mut out_rgb = vec![0u8; w * h];
    unsafe {
      x86_ssse3::bgr_to_luma(
        &mut out_bgr,
        &bgr,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Bgr,
      );
      x86_ssse3::bgr_to_luma(
        &mut out_rgb,
        &rgb,
        w as u32,
        h as u32,
        (w * 3) as u32,
        ChannelOrder::Rgb,
      );
    }
    assert_eq!(out_bgr, out_rgb);
  }

  // ---- clipping_count -------------------------------------------------------

  #[test]
  fn scalar_clipping_count_spot_values() {
    // 3 pixels: clipped-low, unclipped, clipped-high.
    let src = [
      4u8, 0, 0, // max=4, clipped
      100, 100, 100, // max=100, unclipped
      0, 0, 255, // max=255, clipped
    ];
    assert_eq!(
      scalar::Scalar::clipping_count(&src, 3, 1, 9),
      2,
      "expected 2 clipped pixels"
    );
  }

  /// Builds a padded BGR source with distinctive 0xAA padding; pixel
  /// bytes are a fixed-seed random pattern.
  fn assert_clipping_equiv<F: FnOnce(&[u8], u32, u32, u32) -> u64>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut src = vec![0xAAu8; stride * h];
    let mut rng = 0xDEADBEEFu32;
    for y in 0..h {
      for x in 0..w * 3 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        src[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::clipping_count(&src, w as u32, h as u32, stride as u32);
    let simd_out = backend(&src, w as u32, h as u32, stride as u32);
    assert_eq!(
      simd_out, scalar_out,
      "SIMD clipping count disagrees with scalar"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_clipping_count_matches_scalar() {
    // Exercise main loop + tail across a few dims.
    assert_clipping_equiv(16, 4, 16 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(17, 4, 17 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(24, 5, 96, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(257, 31, 257 * 3, |src, w, h, s| unsafe {
      neon::clipping_count(src, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_clipping_count_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_clipping_equiv(16, 4, 16 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(17, 4, 17 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(24, 5, 96, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
    assert_clipping_equiv(257, 31, 257 * 3, |src, w, h, s| unsafe {
      x86_ssse3::clipping_count(src, w, h, s)
    });
  }

  // ---- tenengrad ------------------------------------------------------------

  #[test]
  fn scalar_tenengrad_uniform_is_zero() {
    let data = vec![128u8; 16 * 16];
    assert_eq!(scalar::Scalar::tenengrad(&data, 16, 16, 16), 0.0);
  }

  #[test]
  fn scalar_noise_uniform_is_zero() {
    // No high-frequency signal → σₙ should be 0.
    let data = vec![100u8; 16 * 16];
    assert_eq!(scalar::Scalar::noise(&data, 16, 16, 16), 0.0);
  }

  #[test]
  fn scalar_noise_too_small_is_zero() {
    let data = vec![0u8; 4];
    assert_eq!(scalar::Scalar::noise(&data, 2, 2, 2), 0.0);
  }

  #[test]
  fn scalar_noise_alternating_stripe_matches_closed_form() {
    // Use a checkerboard at amplitude k=64. Every interior pixel sees
    // lap = 4·c - 2·(t+b+l+r) + (tl+tr+bl+br)
    //     = 4·(±64) + 8·(±64) + 4·(±64) = 16·(±64) = ±1024
    // |lap| = 1024 per interior pixel. With interior = 14·14 = 196,
    // mean |lap| = 1024, σₙ ≈ 0.208898 · 1024 ≈ 213.92.
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        let phase = ((x + y) & 1) as i32; // 0 or 1
        let val = 100i32 + if phase == 0 { -64 } else { 64 };
        data[y * w + x] = val as u8;
      }
    }
    let sigma = scalar::Scalar::noise(&data, w, h, w);
    let expected = 0.208_898_754_886_372_3_f64 * 1024.0;
    assert!(
      ((sigma as f64) - expected).abs() < 0.5,
      "expected ~{expected}, got {sigma}"
    );
  }

  #[test]
  fn scalar_noise_stride_padding_is_ignored() {
    // 4 wide × 4 high, stride 8. Padding filled with 255 — would
    // explode |lap| if it leaked into the kernel.
    let w = 4usize;
    let h = 4usize;
    let stride = 8usize;
    let mut data = vec![255u8; stride * h];
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x] = 100; // uniform pixel area → σₙ = 0
      }
    }
    let sigma = scalar::Scalar::noise(&data, w, h, stride);
    assert_eq!(sigma, 0.0, "padding leaked into the Laplacian");
  }

  #[test]
  fn scalar_gradient_anisotropy_zero_mag_is_zero() {
    let (w, h) = (8usize, 8usize);
    let mag = vec![0i32; w * h];
    let dir = vec![0u8; w * h];
    assert_eq!(scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h), 0.0);
  }

  #[test]
  fn scalar_gradient_anisotropy_too_small_is_zero() {
    let mag = vec![100i32; 4];
    let dir = vec![0u8; 4];
    assert_eq!(scalar::Scalar::gradient_anisotropy(&mag, &dir, 2, 2), 0.0);
  }

  #[test]
  fn scalar_gradient_anisotropy_single_direction_is_one() {
    // Every interior pixel has equal magnitude in direction bin 0.
    let (w, h) = (8usize, 8usize);
    let mag = vec![100i32; w * h];
    let dir = vec![0u8; w * h];
    let a = scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h);
    assert!(
      (a - 1.0).abs() < 1e-6,
      "expected 1.0 for single-direction frame, got {a}"
    );
  }

  #[test]
  fn scalar_gradient_anisotropy_uniform_directions_is_zero() {
    // Interior pixels evenly split across all 4 bins → fraction 0.25 → 0.0.
    let (w, h) = (10usize, 10usize);
    let mag = vec![100i32; w * h];
    let mut dir = vec![0u8; w * h];
    // For interior 8×8 = 64 pixels, assign 16 to each of 4 bins.
    let mut counter = 0u8;
    for y in 1..h - 1 {
      for x in 1..w - 1 {
        dir[y * w + x] = counter & 0b11;
        counter = counter.wrapping_add(1);
      }
    }
    let a = scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h);
    assert!(
      a.abs() < 1e-6,
      "expected ~0.0 for uniform directions, got {a}"
    );
    // Sanity-check the construction wrote nonzero magnitudes.
    assert!(mag.iter().any(|&v| v != 0));
  }

  #[test]
  fn scalar_colorfulness_uniform_gray_is_zero() {
    let w = 16usize;
    let h = 16usize;
    let data = vec![128u8; w * h * 3];
    let c = scalar::Scalar::colorfulness(&data, w, h, w * 3, ChannelOrder::Bgr);
    assert!(c.abs() < 1e-3, "expected ~0.0, got {c}");
  }

  #[test]
  fn scalar_colorfulness_pure_red_has_nonzero_score() {
    // Pure red: B=0, G=0, R=255 → rg = 255, yb = 127.5.
    // Constant per pixel, so var_rg = var_yb = 0, σ = 0, μ_rgyb =
    // sqrt(255² + 127.5²) ≈ 285.06.  C = 0 + 0.3 · 285.06 ≈ 85.5.
    let w = 8usize;
    let h = 8usize;
    let mut data = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
      data[i * 3 + 2] = 255;
    }
    let c = scalar::Scalar::colorfulness(&data, w, h, w * 3, ChannelOrder::Bgr);
    let expected = 0.3_f64 * (255.0_f64.powi(2) + 127.5_f64.powi(2)).sqrt();
    assert!(
      ((c as f64) - expected).abs() < 1e-2,
      "expected ~{expected}, got {c}"
    );
  }

  #[test]
  fn scalar_colorfulness_stride_padding_is_ignored() {
    let w = 4usize;
    let h = 4usize;
    let stride = 32usize; // > 4 * 3 = 12
    let mut data = vec![200u8; stride * h]; // padding is "color-y"
    // pixel area: uniform gray
    for y in 0..h {
      for x in 0..w {
        data[y * stride + x * 3] = 128;
        data[y * stride + x * 3 + 1] = 128;
        data[y * stride + x * 3 + 2] = 128;
      }
    }
    let c = scalar::Scalar::colorfulness(&data, w, h, stride, ChannelOrder::Bgr);
    assert!(c.abs() < 1e-3, "padding leaked into reduction, got {c}");
  }

  #[test]
  fn scalar_colorfulness_empty_frame_is_zero() {
    let data = vec![0u8];
    assert_eq!(
      scalar::Scalar::colorfulness(&data, 0, 0, 0, ChannelOrder::Bgr),
      0.0
    );
  }

  #[test]
  fn scalar_tenengrad_small_frame_is_zero() {
    // 2×2 has no interior — no Sobel output.
    let data = vec![0u8, 255, 255, 0];
    assert_eq!(scalar::Scalar::tenengrad(&data, 2, 2, 2), 0.0);
  }

  /// Builds a luma plane with a fixed-seed random pattern; padding is
  /// 0xAA to catch stride misuse.
  fn assert_tenengrad_equiv<F: FnOnce(&[u8], usize, usize, usize) -> f32>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xAAu8; stride * h];
    let mut rng = 0xCAFEBABEu32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::tenengrad(&data, w, h, stride);
    let simd_out = backend(&data, w, h, stride);
    // Tenengrad values vary over many orders of magnitude, but both
    // paths use identical i64 accumulators + one final f64-f32 cast —
    // they should agree to well within 1e-3 relative.
    let rel = (simd_out - scalar_out).abs() / scalar_out.abs().max(1.0);
    assert!(
      rel < 1e-3,
      "SIMD tenengrad disagrees with scalar: simd={simd_out} scalar={scalar_out} rel={rel}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_tenengrad_matches_scalar() {
    // 10 wide = interior 8 wide (one vector iteration, zero tail).
    assert_tenengrad_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // 11 wide = interior 9, one tail pixel per row.
    assert_tenengrad_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // Stride-padded.
    assert_tenengrad_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
    // Larger frame — multiple vector iterations + tail per row.
    assert_tenengrad_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      neon::tenengrad(luma, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_tenengrad_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_tenengrad_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
    assert_tenengrad_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      x86_ssse3::tenengrad(luma, w, h, s)
    });
  }

  // ---- noise ----------------------------------------------------------------

  /// Builds a luma plane with a fixed-seed random pattern; padding is
  /// `0xAA` to catch stride misuse.
  fn assert_noise_equiv<F: FnOnce(&[u8], usize, usize, usize) -> f32>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xAAu8; stride * h];
    let mut rng = 0xC0FFEE_u32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::noise(&data, w, h, stride);
    let simd_out = backend(&data, w, h, stride);
    // Both paths use identical i64 accumulators and the same f64
    // `COEFF * Σ / interior` scaling, so f32 agreement should be
    // tight.
    let rel = (simd_out - scalar_out).abs() / scalar_out.abs().max(1.0);
    assert!(
      rel < 1e-4,
      "SIMD noise disagrees with scalar: simd={simd_out} scalar={scalar_out} rel={rel}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_noise_matches_scalar() {
    // 10 wide → interior 8 wide (one vector iteration, zero tail).
    assert_noise_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      neon::noise(luma, w, h, s)
    });
    // 11 wide → interior 9, one tail pixel per row.
    assert_noise_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      neon::noise(luma, w, h, s)
    });
    // Stride-padded — sentinel 0xAA in the padding must not leak.
    assert_noise_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      neon::noise(luma, w, h, s)
    });
    // Larger frame — multiple vector iterations + tail per row.
    assert_noise_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      neon::noise(luma, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_noise_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_noise_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      x86_ssse3::noise(luma, w, h, s)
    });
    assert_noise_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      x86_ssse3::noise(luma, w, h, s)
    });
    assert_noise_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      x86_ssse3::noise(luma, w, h, s)
    });
    assert_noise_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      x86_ssse3::noise(luma, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn sse41_noise_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    assert_noise_equiv(10, 10, 10, |luma, w, h, s| unsafe {
      x86_sse41::noise(luma, w, h, s)
    });
    assert_noise_equiv(11, 10, 11, |luma, w, h, s| unsafe {
      x86_sse41::noise(luma, w, h, s)
    });
    assert_noise_equiv(24, 12, 64, |luma, w, h, s| unsafe {
      x86_sse41::noise(luma, w, h, s)
    });
    assert_noise_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      x86_sse41::noise(luma, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_noise_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    // 18 wide → interior 16 → one 16-lane vector iteration, zero tail.
    assert_noise_equiv(18, 10, 18, |luma, w, h, s| unsafe {
      x86_avx2::noise(luma, w, h, s)
    });
    // 19 wide → interior 17, one tail pixel per row.
    assert_noise_equiv(19, 10, 19, |luma, w, h, s| unsafe {
      x86_avx2::noise(luma, w, h, s)
    });
    // Stride-padded.
    assert_noise_equiv(40, 12, 96, |luma, w, h, s| unsafe {
      x86_avx2::noise(luma, w, h, s)
    });
    // Larger frame — multiple chunks + tail.
    assert_noise_equiv(257, 31, 257, |luma, w, h, s| unsafe {
      x86_avx2::noise(luma, w, h, s)
    });
  }

  // ---- colorfulness ---------------------------------------------------------

  /// Builds a packed BGR plane with a fixed-seed random pattern.
  /// Padding bytes (between `w*3` and `stride`) are set to `0xCC`
  /// so a SIMD load that reads past the pixel area trips the
  /// equivalence assertion.
  fn assert_colorfulness_equiv<F: FnOnce(&[u8], usize, usize, usize) -> f32>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xCCu8; stride * h];
    let mut rng = 0xDEAD_BEEFu32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let base = y * stride + x * 3;
        data[base] = (rng >> 24) as u8;
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[base + 1] = (rng >> 24) as u8;
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[base + 2] = (rng >> 24) as u8;
      }
    }
    let scalar_out = scalar::Scalar::colorfulness(&data, w, h, stride, ChannelOrder::Bgr);
    let simd_out = backend(&data, w, h, stride);
    // The integer two-pass formulation `E[X²] - E[X]²` produces
    // results that differ from Welford only by f64 rounding, well
    // inside f32 precision for our value ranges.
    let rel = (simd_out - scalar_out).abs() / scalar_out.abs().max(1.0);
    assert!(
      rel < 1e-4,
      "SIMD colorfulness disagrees with scalar: simd={simd_out} scalar={scalar_out} rel={rel}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_colorfulness_matches_scalar() {
    // 16 wide = one vector iteration, zero tail.
    assert_colorfulness_equiv(16, 12, 48, |bgr, w, h, s| unsafe {
      neon::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    // 17 wide → 1 chunk + 1 tail pixel.
    assert_colorfulness_equiv(17, 12, 51, |bgr, w, h, s| unsafe {
      neon::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    // Stride-padded (sentinel 0xCC in padding must not leak).
    assert_colorfulness_equiv(24, 9, 128, |bgr, w, h, s| unsafe {
      neon::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    // Larger frame — multiple chunks + tail per row.
    assert_colorfulness_equiv(259, 17, 259 * 3, |bgr, w, h, s| unsafe {
      neon::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_colorfulness_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_colorfulness_equiv(16, 12, 48, |bgr, w, h, s| unsafe {
      x86_ssse3::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(17, 12, 51, |bgr, w, h, s| unsafe {
      x86_ssse3::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(24, 9, 128, |bgr, w, h, s| unsafe {
      x86_ssse3::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(259, 17, 259 * 3, |bgr, w, h, s| unsafe {
      x86_ssse3::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn sse41_colorfulness_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    assert_colorfulness_equiv(16, 12, 48, |bgr, w, h, s| unsafe {
      x86_sse41::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(17, 12, 51, |bgr, w, h, s| unsafe {
      x86_sse41::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(24, 9, 128, |bgr, w, h, s| unsafe {
      x86_sse41::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(259, 17, 259 * 3, |bgr, w, h, s| unsafe {
      x86_sse41::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_colorfulness_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    assert_colorfulness_equiv(16, 12, 48, |bgr, w, h, s| unsafe {
      x86_avx2::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(17, 12, 51, |bgr, w, h, s| unsafe {
      x86_avx2::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(24, 9, 128, |bgr, w, h, s| unsafe {
      x86_avx2::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
    assert_colorfulness_equiv(259, 17, 259 * 3, |bgr, w, h, s| unsafe {
      x86_avx2::colorfulness(bgr, w, h, s, ChannelOrder::Bgr)
    });
  }

  // ---- gradient_anisotropy --------------------------------------------------

  /// Builds `(mag, dir)` planes with a fixed-seed pattern that
  /// exercises every code path: ~25% negative `mag` (skipped), all
  /// four direction bins, and a mix of magnitudes.
  fn assert_anisotropy_equiv<F: FnOnce(&[i32], &[u8], usize, usize) -> f32>(
    w: usize,
    h: usize,
    backend: F,
  ) {
    let mut mag = vec![0i32; w * h];
    let mut dir = vec![0u8; w * h];
    let mut rng = 0xFEED_FACEu32;
    for v in mag.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      // Span `i32::MIN/4..i32::MAX/4` so ~half the lanes hit the
      // `mag <= 0` skip branch and the rest contribute large
      // positive magnitudes.
      *v = (rng as i32) / 4;
    }
    for v in dir.iter_mut() {
      rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (rng >> 24) as u8;
    }
    let scalar_out = scalar::Scalar::gradient_anisotropy(&mag, &dir, w, h);
    let simd_out = backend(&mag, &dir, w, h);
    let rel = (simd_out - scalar_out).abs() / scalar_out.abs().max(1.0);
    assert!(
      rel < 1e-5,
      "SIMD gradient_anisotropy disagrees with scalar: simd={simd_out} scalar={scalar_out} rel={rel}"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_gradient_anisotropy_matches_scalar() {
    // 10 wide → interior 8 wide → one 8-lane vector iteration,
    // zero tail.
    assert_anisotropy_equiv(10, 10, |m, d, w, h| unsafe {
      neon::gradient_anisotropy(m, d, w, h)
    });
    // 11 wide → interior 9 → one chunk + 1 tail pixel per row.
    assert_anisotropy_equiv(11, 10, |m, d, w, h| unsafe {
      neon::gradient_anisotropy(m, d, w, h)
    });
    // Larger frame — multiple chunks + tail.
    assert_anisotropy_equiv(257, 31, |m, d, w, h| unsafe {
      neon::gradient_anisotropy(m, d, w, h)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_gradient_anisotropy_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    // SSSE3 uses 4-pixel chunks.
    assert_anisotropy_equiv(6, 6, |m, d, w, h| unsafe {
      x86_ssse3::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(7, 6, |m, d, w, h| unsafe {
      x86_ssse3::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(11, 10, |m, d, w, h| unsafe {
      x86_ssse3::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(257, 31, |m, d, w, h| unsafe {
      x86_ssse3::gradient_anisotropy(m, d, w, h)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn sse41_gradient_anisotropy_matches_scalar() {
    if !std::is_x86_feature_detected!("sse4.1") {
      return;
    }
    assert_anisotropy_equiv(6, 6, |m, d, w, h| unsafe {
      x86_sse41::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(7, 6, |m, d, w, h| unsafe {
      x86_sse41::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(11, 10, |m, d, w, h| unsafe {
      x86_sse41::gradient_anisotropy(m, d, w, h)
    });
    assert_anisotropy_equiv(257, 31, |m, d, w, h| unsafe {
      x86_sse41::gradient_anisotropy(m, d, w, h)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn avx2_gradient_anisotropy_matches_scalar() {
    if !std::is_x86_feature_detected!("avx2") {
      return;
    }
    // 10 wide → interior 8 → one 8-lane vector iteration, zero tail.
    assert_anisotropy_equiv(10, 10, |m, d, w, h| unsafe {
      x86_avx2::gradient_anisotropy(m, d, w, h)
    });
    // 11 wide → interior 9, one tail pixel per row.
    assert_anisotropy_equiv(11, 10, |m, d, w, h| unsafe {
      x86_avx2::gradient_anisotropy(m, d, w, h)
    });
    // Larger frame — multiple chunks + tail.
    assert_anisotropy_equiv(257, 31, |m, d, w, h| unsafe {
      x86_avx2::gradient_anisotropy(m, d, w, h)
    });
  }

  #[test]
  #[should_panic(expected = "mag.len()")]
  fn gradient_anisotropy_dispatcher_panics_on_undersized_mag() {
    // The SIMD backends use raw pointer loads — passing a `mag`
    // slice shorter than `w*h` would be UB. The dispatcher MUST
    // panic before dispatching to either path.
    let mag = vec![0i32; 10]; // 10 < 8*8 = 64
    let dir = vec![0u8; 64];
    let _ = gradient_anisotropy(&mag, &dir, 8, 8, true);
  }

  #[test]
  #[should_panic(expected = "dir.len()")]
  fn gradient_anisotropy_dispatcher_panics_on_undersized_dir() {
    let mag = vec![0i32; 64];
    let dir = vec![0u8; 10];
    let _ = gradient_anisotropy(&mag, &dir, 8, 8, true);
  }

  #[test]
  #[should_panic(expected = "mag.len()")]
  fn gradient_anisotropy_dispatcher_panics_even_on_scalar_path() {
    // The bounds check fires before the `use_simd` branch so the
    // scalar caller gets the same explicit diagnostic instead of
    // an `index out of bounds` panic deep inside the kernel.
    let mag = vec![0i32; 10];
    let dir = vec![0u8; 64];
    let _ = gradient_anisotropy(&mag, &dir, 8, 8, false);
  }

  // ---- plane_mean_variance --------------------------------------------------

  #[test]
  fn scalar_plane_mean_variance_uniform() {
    let data = vec![128u8; 16 * 8];
    let (mean, var) = scalar::Scalar::plane_mean_variance(&data, 16, 8, 16);
    assert!((mean - 128.0).abs() < 1e-3);
    assert!(var < 1e-3);
  }

  #[test]
  fn scalar_plane_mean_variance_bimodal() {
    // Half zeros, half 200 → mean 100, var 10 000.
    let mut data = vec![0u8; 16 * 16];
    for i in (16 * 8)..(16 * 16) {
      data[i] = 200;
    }
    let (mean, var) = scalar::Scalar::plane_mean_variance(&data, 16, 16, 16);
    assert!((mean - 100.0).abs() < 1e-3);
    assert!((var - 10_000.0).abs() < 1.0);
  }

  fn assert_plane_stats_equiv<F: FnOnce(&[u8], usize, usize, usize) -> (f32, f32)>(
    w: usize,
    h: usize,
    stride: usize,
    backend: F,
  ) {
    let mut data = vec![0xAAu8; stride * h];
    let mut rng = 0xABAD1DEAu32;
    for y in 0..h {
      for x in 0..w {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        data[y * stride + x] = (rng >> 24) as u8;
      }
    }
    let (scalar_m, scalar_v) = scalar::Scalar::plane_mean_variance(&data, w, h, stride);
    let (simd_m, simd_v) = backend(&data, w, h, stride);
    let rel_m = (simd_m - scalar_m).abs() / scalar_m.abs().max(1.0);
    let rel_v = (simd_v - scalar_v).abs() / scalar_v.abs().max(1.0);
    assert!(
      rel_m < 1e-5 && rel_v < 1e-4,
      "plane stats disagree: mean simd={simd_m} scalar={scalar_m} (rel {rel_m}), \
       var simd={simd_v} scalar={scalar_v} (rel {rel_v})"
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_plane_mean_variance_matches_scalar() {
    assert_plane_stats_equiv(16, 4, 16, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(17, 4, 17, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(24, 5, 64, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(257, 31, 257, |p, w, h, s| unsafe {
      neon::plane_mean_variance(p, w, h, s)
    });
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_plane_mean_variance_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    assert_plane_stats_equiv(16, 4, 16, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(17, 4, 17, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(24, 5, 64, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
    assert_plane_stats_equiv(257, 31, 257, |p, w, h, s| unsafe {
      x86_ssse3::plane_mean_variance(p, w, h, s)
    });
  }

  // Regression guard for the SSSE3 plane_mean_variance squared-sum
  // truncation bug: when v ≥ 182, v² ≥ 33 124 overflows i16. A backend
  // that computes the square as u16 first and then feeds the result
  // through `_mm_madd_epi16` (treating u16 as signed) reports wildly
  // wrong variance — non-zero on a uniform plane, or sign-flipped on
  // a bimodal one.
  //
  // The scalar reference always handled this correctly; this test
  // pins the invariant so any SIMD backend that violates it fails
  // CI. See also `ssse3_plane_mean_variance_bright_matches_scalar`.
  #[test]
  fn scalar_plane_mean_variance_bright_uniform_is_zero_var() {
    for &v in &[182u8, 200, 240, 255] {
      let data = vec![v; 32 * 8];
      let (mean, var) = scalar::Scalar::plane_mean_variance(&data, 32, 8, 32);
      assert!(
        (mean - v as f32).abs() < 1e-3,
        "uniform v={v}: mean {mean} ≠ {v}"
      );
      assert!(var < 1e-3, "uniform v={v}: variance {var} should be 0");
    }
  }

  #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
  #[test]
  fn ssse3_plane_mean_variance_bright_matches_scalar() {
    if !std::is_x86_feature_detected!("ssse3") {
      return;
    }
    // Bright uniform plane: variance must be ~0. The SSSE3 backend's
    // historical bug squared u8 lanes at u16 precision before widening,
    // producing huge negative-then-zero-extended variance on values
    // ≥ 182.
    for &v in &[182u8, 200, 240, 255] {
      let data = vec![v; 32 * 8];
      let (scalar_m, scalar_v) = scalar::Scalar::plane_mean_variance(&data, 32, 8, 32);
      let (simd_m, simd_v) = unsafe { x86_ssse3::plane_mean_variance(&data, 32, 8, 32) };
      assert!(
        (simd_m - scalar_m).abs() < 1e-3,
        "v={v}: mean simd={simd_m} scalar={scalar_m}"
      );
      assert!(
        (simd_v - scalar_v).abs() < 1e-3,
        "v={v}: variance simd={simd_v} scalar={scalar_v}"
      );
    }
  }
}
