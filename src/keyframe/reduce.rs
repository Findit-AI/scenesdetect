//! Scalar planar reductions shared across the keyframe metric detectors.
//!
//! Kept in its own module so a future SIMD backend has a single
//! reduction kernel to port — the luma and saturation variance paths
//! are otherwise a standing invitation for drift between two nearly
//! identical scalar loops.

/// Single-pass mean + population variance of a single-plane `u8` image,
/// routed through [`crate::arch::plane_mean_variance`] for SIMD
/// dispatch.
///
/// Row stride is honoured: only the first `width` bytes of each row
/// participate. `width * height == 0` yields `(0.0, 0.0)`.
///
/// Returns `(mean, variance)` both in 0-255 space (the input scale),
/// as `f32` for downstream consumption.
#[inline]
pub(super) fn plane_mean_variance(
  plane: &[u8],
  width: usize,
  height: usize,
  stride: usize,
  use_simd: bool,
) -> (f32, f32) {
  crate::arch::plane_mean_variance(plane, width, height, stride, use_simd)
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use std::vec;

  #[test]
  fn black_yields_zero_zero() {
    let data = vec![0u8; 16 * 8];
    assert_eq!(plane_mean_variance(&data, 16, 8, 16, true), (0.0, 0.0));
  }

  #[test]
  fn uniform_gray_has_mean_and_zero_variance() {
    let data = vec![128u8; 16 * 8];
    let (mean, var) = plane_mean_variance(&data, 16, 8, 16, true);
    assert!((mean - 128.0).abs() < 1e-3);
    assert!(var < 1e-3);
  }

  #[test]
  fn bimodal_matches_expected_variance() {
    // Half zeros, half 200 → mean = 100, var = 100² = 10_000.
    let mut data = vec![0u8; 16 * 16];
    for v in data.iter_mut().take(16 * 16).skip(16 * 8) {
      *v = 200;
    }
    let (mean, var) = plane_mean_variance(&data, 16, 16, 16, true);
    assert!((mean - 100.0).abs() < 1e-3);
    assert!((var - 10_000.0).abs() < 1.0);
  }

  #[test]
  fn stride_padding_is_ignored() {
    // 4×2 pixels at stride 8. Padding set to 255; pixel data to 0.
    let stride = 8usize;
    let mut data = vec![255u8; stride * 2];
    for y in 0..2 {
      for x in 0..4 {
        data[y * stride + x] = 0;
      }
    }
    let (mean, var) = plane_mean_variance(&data, 4, 2, stride, true);
    assert_eq!(mean, 0.0);
    assert_eq!(var, 0.0);
  }

  #[test]
  fn zero_dim_returns_zeros() {
    let data = vec![100u8; 16];
    assert_eq!(plane_mean_variance(&data, 0, 10, 16, true), (0.0, 0.0));
    assert_eq!(plane_mean_variance(&data, 10, 0, 16, true), (0.0, 0.0));
  }
}
