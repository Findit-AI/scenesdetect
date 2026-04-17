//! x86 / x86_64 AVX2 backend for BGR→HSV.
//!
//! Processes 16 pixels per iteration, same as SSSE3, but performs the HSV
//! arithmetic on `__m256` (8-wide f32) in two groups of 8 pixels — half as
//! many arithmetic passes as SSSE3. The deinterleave still uses SSSE3-style
//! `_mm_shuffle_epi8` inside 128-bit lanes (AVX2's 32-pixel-wide deinterleave
//! needs cross-lane permutes; that's a meaningful complexity jump for modest
//! extra throughput on this workload).
//!
//! Gated on the `avx2` target feature. The dispatcher in
//! [`super::bgr_to_hsv_planes`] picks this backend only when
//! `is_x86_feature_detected!("avx2")` at runtime (or `target_feature = "avx2"`
//! at compile time in no_std builds).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Same PSHUFB masks as the SSSE3 backend (see `x86_ssse3` for comments).

const BLK0_B: [i8; 16] = [0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK0_G: [i8; 16] = [1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK0_R: [i8; 16] = [2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
const BLK1_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1];
const BLK1_G: [i8; 16] = [-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1];
const BLK1_R: [i8; 16] = [-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1];
const BLK2_B: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13];
const BLK2_G: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14];
const BLK2_R: [i8; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15];

/// AVX2 BGR→HSV: 16 pixels per iteration, 8-wide HSV arithmetic.
///
/// # Safety
///
/// Caller must ensure AVX2 (which implies SSSE3) is available.
#[target_feature(enable = "avx2", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_hsv_planes(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  src: &[u8],
  width: u32,
  height: u32,
  stride: u32,
) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let m_b0 = unsafe { _mm_loadu_si128(BLK0_B.as_ptr() as *const __m128i) };
  let m_g0 = unsafe { _mm_loadu_si128(BLK0_G.as_ptr() as *const __m128i) };
  let m_r0 = unsafe { _mm_loadu_si128(BLK0_R.as_ptr() as *const __m128i) };
  let m_b1 = unsafe { _mm_loadu_si128(BLK1_B.as_ptr() as *const __m128i) };
  let m_g1 = unsafe { _mm_loadu_si128(BLK1_G.as_ptr() as *const __m128i) };
  let m_r1 = unsafe { _mm_loadu_si128(BLK1_R.as_ptr() as *const __m128i) };
  let m_b2 = unsafe { _mm_loadu_si128(BLK2_B.as_ptr() as *const __m128i) };
  let m_g2 = unsafe { _mm_loadu_si128(BLK2_G.as_ptr() as *const __m128i) };
  let m_r2 = unsafe { _mm_loadu_si128(BLK2_R.as_ptr() as *const __m128i) };
  let zero_i = unsafe { _mm_setzero_si128() };

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { _mm_loadu_si128(p as *const __m128i) };
      let blk1 = unsafe { _mm_loadu_si128(p.add(16) as *const __m128i) };
      let blk2 = unsafe { _mm_loadu_si128(p.add(32) as *const __m128i) };

      let b = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_b0), _mm_shuffle_epi8(blk1, m_b1)),
          _mm_shuffle_epi8(blk2, m_b2),
        )
      };
      let g = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_g0), _mm_shuffle_epi8(blk1, m_g1)),
          _mm_shuffle_epi8(blk2, m_g2),
        )
      };
      let r = unsafe {
        _mm_or_si128(
          _mm_or_si128(_mm_shuffle_epi8(blk0, m_r0), _mm_shuffle_epi8(blk1, m_r1)),
          _mm_shuffle_epi8(blk2, m_r2),
        )
      };

      // Widen u8x16 → u32x8 (low 8 pixels, high 8 pixels) → f32x8 per channel.
      //   _mm256_cvtepu8_epi32 takes the low 8 bytes of an __m128i.
      let b_lo32 = unsafe { _mm256_cvtepu8_epi32(b) };
      let b_hi32 = unsafe { _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(b, b)) };
      let g_lo32 = unsafe { _mm256_cvtepu8_epi32(g) };
      let g_hi32 = unsafe { _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(g, g)) };
      let r_lo32 = unsafe { _mm256_cvtepu8_epi32(r) };
      let r_hi32 = unsafe { _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(r, r)) };

      let b_lo = unsafe { _mm256_cvtepi32_ps(b_lo32) };
      let b_hi = unsafe { _mm256_cvtepi32_ps(b_hi32) };
      let g_lo = unsafe { _mm256_cvtepi32_ps(g_lo32) };
      let g_hi = unsafe { _mm256_cvtepi32_ps(g_hi32) };
      let r_lo = unsafe { _mm256_cvtepi32_ps(r_lo32) };
      let r_hi = unsafe { _mm256_cvtepi32_ps(r_hi32) };

      let (hue_lo, sat_lo, val_lo) = unsafe { bgr_to_hsv_f32x8(b_lo, g_lo, r_lo) };
      let (hue_hi, sat_hi, val_hi) = unsafe { bgr_to_hsv_f32x8(b_hi, g_hi, r_hi) };

      // Hue/2 → i32, clamp [0, 179]; S, V → i32, clamp [0, 255].
      // Use add-0.5 + truncate (round half-up for non-negative values) to
      // match the scalar `round()` semantics instead of MXCSR's default
      // round-to-nearest-even via `_mm256_cvtps_epi32`.
      let half = unsafe { _mm256_set1_ps(0.5) };
      let round_half = half; // reuse for the add-then-truncate pattern
      let hh_lo_i =
        unsafe { _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(hue_lo, half), round_half)) };
      let hh_hi_i =
        unsafe { _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(hue_hi, half), round_half)) };
      let ss_lo_i = unsafe { _mm256_cvttps_epi32(_mm256_add_ps(sat_lo, round_half)) };
      let ss_hi_i = unsafe { _mm256_cvttps_epi32(_mm256_add_ps(sat_hi, round_half)) };
      let vv_lo_i = unsafe { _mm256_cvttps_epi32(_mm256_add_ps(val_lo, round_half)) };
      let vv_hi_i = unsafe { _mm256_cvttps_epi32(_mm256_add_ps(val_hi, round_half)) };

      let h_lo = unsafe { _mm256_min_epi32(hh_lo_i, _mm256_set1_epi32(179)) };
      let h_hi = unsafe { _mm256_min_epi32(hh_hi_i, _mm256_set1_epi32(179)) };
      let s_lo = unsafe { _mm256_min_epi32(ss_lo_i, _mm256_set1_epi32(255)) };
      let s_hi = unsafe { _mm256_min_epi32(ss_hi_i, _mm256_set1_epi32(255)) };
      let v_lo = unsafe { _mm256_min_epi32(vv_lo_i, _mm256_set1_epi32(255)) };
      let v_hi = unsafe { _mm256_min_epi32(vv_hi_i, _mm256_set1_epi32(255)) };

      let h_vec = unsafe { pack_avx2(h_lo, h_hi) };
      let s_vec = unsafe { pack_avx2(s_lo, s_hi) };
      let v_vec = unsafe { pack_avx2(v_lo, v_hi) };

      unsafe {
        _mm_storeu_si128(h_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, h_vec);
        _mm_storeu_si128(s_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, s_vec);
        _mm_storeu_si128(v_out.as_mut_ptr().add(dst_off + x) as *mut __m128i, v_vec);
      }

      x += LANES;
    }

    // Scalar tail. Silence unused warning if the block is fully consumed.
    let _ = zero_i;
    let row = &src[row_base..row_base + w * 3];
    while x < w {
      let b = row[x * 3] as f32;
      let g = row[x * 3 + 1] as f32;
      let r = row[x * 3 + 2] as f32;
      let (hue, sat, val) = super::scalar::Scalar::bgr_to_hsv_pixel(b, g, r);
      h_out[dst_off + x] = hue;
      s_out[dst_off + x] = sat;
      v_out[dst_off + x] = val;
      x += 1;
    }
  }
}

/// Pack two `i32x8` vectors (values ≤ 255) into one `u8x16`.
///
/// `_mm256_packs_epi32` packs *within 128-bit lanes*, so the result needs a
/// `_mm256_permute4x64_epi64` to reorder lanes into sequential order.
#[target_feature(enable = "avx2")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn pack_avx2(lo: __m256i, hi: __m256i) -> __m128i {
  // i32x8 + i32x8 → i16x16 with per-128-bit-lane pack: layout
  //   [lo[0..4], hi[0..4], lo[4..8], hi[4..8]]
  let packed16 = unsafe { _mm256_packs_epi32(lo, hi) };
  // Reorder to [lo[0..4], lo[4..8], hi[0..4], hi[4..8]] so the 8 lo values
  // and 8 hi values sit in separate 128-bit halves.
  let reordered = unsafe { _mm256_permute4x64_epi64::<0b1101_1000>(packed16) };
  // i16x16 → u8x16: packus saturates per 128-bit lane. After the permute,
  // lanes are ordered such that packing the two halves together gives the
  // right sequential layout.
  let packed8 = unsafe { _mm256_packus_epi16(reordered, reordered) };
  // Extract the low 128 bits (both halves are duplicates after packus).
  unsafe { _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1101_1000>(packed8)) }
}

/// Branch-free 8-lane BGR→HSV core. Same algorithm as NEON / SSSE3, AVX
/// intrinsics.
#[target_feature(enable = "avx2")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn bgr_to_hsv_f32x8(b: __m256, g: __m256, r: __m256) -> (__m256, __m256, __m256) {
  let zero = unsafe { _mm256_setzero_ps() };
  let one = unsafe { _mm256_set1_ps(1.0) };

  let v = unsafe { _mm256_max_ps(_mm256_max_ps(b, g), r) };
  let min = unsafe { _mm256_min_ps(_mm256_min_ps(b, g), r) };
  let delta = unsafe { _mm256_sub_ps(v, min) };

  let delta_zero = unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(delta, zero) };
  let v_zero = unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(v, zero) };
  let delta_safe = unsafe { _mm256_blendv_ps(delta, one, delta_zero) };

  let sixty = unsafe { _mm256_set1_ps(60.0) };
  let c120 = unsafe { _mm256_set1_ps(120.0) };
  let c240 = unsafe { _mm256_set1_ps(240.0) };
  let c360 = unsafe { _mm256_set1_ps(360.0) };
  let c255 = unsafe { _mm256_set1_ps(255.0) };

  let h_r = unsafe { _mm256_div_ps(_mm256_mul_ps(sixty, _mm256_sub_ps(g, b)), delta_safe) };
  let h_g = unsafe {
    _mm256_add_ps(
      _mm256_div_ps(_mm256_mul_ps(sixty, _mm256_sub_ps(b, r)), delta_safe),
      c120,
    )
  };
  let h_b = unsafe {
    _mm256_add_ps(
      _mm256_div_ps(_mm256_mul_ps(sixty, _mm256_sub_ps(r, g)), delta_safe),
      c240,
    )
  };

  let is_r = unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(v, r) };
  let is_g = unsafe { _mm256_cmp_ps::<_CMP_EQ_OQ>(v, g) };
  let not_r_and_g = unsafe { _mm256_andnot_ps(is_r, is_g) };
  let hue_rg = unsafe { _mm256_blendv_ps(h_b, h_r, is_r) };
  let hue = unsafe { _mm256_blendv_ps(hue_rg, h_g, not_r_and_g) };
  let neg = unsafe { _mm256_cmp_ps::<_CMP_LT_OQ>(hue, zero) };
  let hue = unsafe { _mm256_blendv_ps(hue, _mm256_add_ps(hue, c360), neg) };
  let hue = unsafe { _mm256_blendv_ps(hue, zero, delta_zero) };

  let v_safe = unsafe { _mm256_blendv_ps(v, one, v_zero) };
  let sat = unsafe { _mm256_div_ps(_mm256_mul_ps(c255, delta), v_safe) };
  let sat = unsafe { _mm256_blendv_ps(sat, zero, v_zero) };

  (hue, sat, v)
}
