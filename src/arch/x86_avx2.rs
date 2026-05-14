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

/// AVX2 Immerkaer noise estimator on a u8 luma plane.
///
/// 16 pixels per chunk — double the SSSE3 backend's 8-lane
/// width by widening directly to `i16×16` in `__m256i` via
/// `_mm256_cvtepu8_epi16`.
///
/// Per 16-pixel chunk:
/// - Load 9 `u8×16` neighborhoods (`tl…br`) via
///   `_mm_loadu_si128`.
/// - `_mm256_cvtepu8_epi16` widens each to `i16×16`.
/// - `lap = 4·c - 2·(t+b+l+r) + (tl+tr+bl+br)` lanewise.
///   Peak `|lap| = 16·255 = 4080` fits in `i16` (max 32767).
/// - `_mm256_abs_epi16` → per-pixel absolutes.
/// - `_mm256_madd_epi16(abs, ones16)` pair-sums into `i32×8`
///   (lane peak `2·4080 = 8160`).
/// - `_mm256_cvtepi32_epi64` widens each 128-bit half of the
///   `i32×8` into `i64×4` (sign-extend; values non-negative)
///   and accumulates into an `i64×4` accumulator — keeps the
///   running sum within `i64::MAX` for any frame size.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unused_unsafe)]
pub(super) unsafe fn noise(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
  if w < 3 || h < 3 {
    return 0.0;
  }
  let interior = (w - 2) * (h - 2);
  if interior == 0 {
    return 0.0;
  }

  const LANES: usize = 16;
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  let ones16 = unsafe { _mm256_set1_epi16(1) };
  let mut acc = unsafe { _mm256_setzero_si256() }; // i64×4
  let mut tail_acc: i64 = 0;

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      let load16 = |p: *const u8| -> __m128i { unsafe { _mm_loadu_si128(p as *const __m128i) } };

      let tl = load16(unsafe { prev.as_ptr().add(x - 1) });
      let t = load16(unsafe { prev.as_ptr().add(x) });
      let tr = load16(unsafe { prev.as_ptr().add(x + 1) });
      let l = load16(unsafe { curr.as_ptr().add(x - 1) });
      let c = load16(unsafe { curr.as_ptr().add(x) });
      let r = load16(unsafe { curr.as_ptr().add(x + 1) });
      let bl = load16(unsafe { next.as_ptr().add(x - 1) });
      let b = load16(unsafe { next.as_ptr().add(x) });
      let br = load16(unsafe { next.as_ptr().add(x + 1) });

      // Zero-extend u8×16 → i16×16 in __m256i.
      let tl = unsafe { _mm256_cvtepu8_epi16(tl) };
      let t = unsafe { _mm256_cvtepu8_epi16(t) };
      let tr = unsafe { _mm256_cvtepu8_epi16(tr) };
      let l = unsafe { _mm256_cvtepu8_epi16(l) };
      let c = unsafe { _mm256_cvtepu8_epi16(c) };
      let r = unsafe { _mm256_cvtepu8_epi16(r) };
      let bl = unsafe { _mm256_cvtepu8_epi16(bl) };
      let b = unsafe { _mm256_cvtepu8_epi16(b) };
      let br = unsafe { _mm256_cvtepu8_epi16(br) };

      // lap = 4c - 2(t + b + l + r) + (tl + tr + bl + br)
      let four_c = unsafe { _mm256_slli_epi16::<2>(c) };
      let tblr = unsafe { _mm256_add_epi16(_mm256_add_epi16(t, b), _mm256_add_epi16(l, r)) };
      let two_tblr = unsafe { _mm256_slli_epi16::<1>(tblr) };
      let corners = unsafe { _mm256_add_epi16(_mm256_add_epi16(tl, tr), _mm256_add_epi16(bl, br)) };
      let lap = unsafe { _mm256_add_epi16(_mm256_sub_epi16(four_c, two_tblr), corners) };
      let abs_lap = unsafe { _mm256_abs_epi16(lap) };

      // Pair-sum i16×16 → i32×8.
      let pairs = unsafe { _mm256_madd_epi16(abs_lap, ones16) };

      // Widen i32×8 → i64×4 × 2 (non-negative inputs).
      let lo_128 = unsafe { _mm256_castsi256_si128(pairs) };
      let hi_128 = unsafe { _mm256_extracti128_si256::<1>(pairs) };
      let lo64 = unsafe { _mm256_cvtepi32_epi64(lo_128) };
      let hi64 = unsafe { _mm256_cvtepi32_epi64(hi_128) };
      acc = unsafe { _mm256_add_epi64(acc, lo64) };
      acc = unsafe { _mm256_add_epi64(acc, hi64) };

      x += LANES;
    }

    while x < w - 1 {
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
      let lap = 4 * c - 2 * (t + b + l + r) + (tl + tr + bl + br);
      tail_acc += lap.unsigned_abs() as i64;
      x += 1;
    }
  }

  // Reduce i64×4 → i64.
  let mut lanes = [0i64; 4];
  unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc) };
  let vec_sum: i64 = lanes.iter().sum();

  const COEFF: f64 = 0.208_898_754_886_372_3;
  (((vec_sum + tail_acc) as f64) * COEFF / (interior as f64)) as f32
}

/// AVX2 Hasler-Süßstrunk colourfulness on packed 24-bit BGR.
///
/// 16-pixel chunks: the deinterleave still uses 128-bit SSSE3
/// `pshufb` (cross-lane permutes for AVX2-wide deinterleave
/// aren't worth the complexity here), but the per-pixel
/// arithmetic runs on `i16×16` in a single `__m256i`, halving
/// the per-chunk operation count vs the SSSE3 backend's
/// low/high-half pair.
///
/// Per chunk:
/// - 3 × 16-byte loads + 9 `pshufb` shuffles → `b`, `g`, `r` as
///   `u8×16` (same deinterleave table as `bgr_to_hsv_planes`).
/// - `_mm256_cvtepu8_epi16` widens each to `i16×16`.
/// - `rg = R - G`, `u = R + G - 2B` lanewise.
/// - `_mm256_madd_epi16(v, ones16)` pair-sums to `i32×8` for
///   `Σ rg` / `Σ u`; `_mm256_madd_epi16(v, v)` for `Σ rg²` /
///   `Σ u²`. Each `i32×8` is split via
///   `_mm256_castsi256_si128` / `_mm256_extracti128_si256` and
///   widened to `i64×4 × 2` via `_mm256_cvtepi32_epi64`
///   (sign-extend; squares are non-negative so sign and zero
///   extension coincide for that stream). Accumulators are
///   `i64×4`, so no frame size can wrap them.
///
/// # Safety
///
/// Caller must ensure AVX2 (which implies SSSE3) is available.
#[target_feature(enable = "avx2", enable = "ssse3")]
#[allow(unused_unsafe)]
pub(super) unsafe fn colorfulness(bgr: &[u8], w: usize, h: usize, stride: usize) -> f32 {
  let n = w.saturating_mul(h);
  if n == 0 {
    return 0.0;
  }

  const LANES: usize = 16;
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
  let ones16 = unsafe { _mm256_set1_epi16(1) };

  let mut sum_rg = unsafe { _mm256_setzero_si256() }; // i64×4
  let mut sum_u = unsafe { _mm256_setzero_si256() }; // i64×4
  let mut sum_rg_sq = unsafe { _mm256_setzero_si256() }; // i64×4 (non-neg)
  let mut sum_u_sq = unsafe { _mm256_setzero_si256() }; // i64×4 (non-neg)
  let mut tail_sum_rg: i64 = 0;
  let mut tail_sum_u: i64 = 0;
  let mut tail_sum_rg_sq: u64 = 0;
  let mut tail_sum_u_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * stride;

    let mut x = 0;
    while x < whole {
      let p = unsafe { bgr.as_ptr().add(row_base + x * 3) };
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

      // Widen u8×16 → i16×16 in __m256i.
      let b16 = unsafe { _mm256_cvtepu8_epi16(b) };
      let g16 = unsafe { _mm256_cvtepu8_epi16(g) };
      let r16 = unsafe { _mm256_cvtepu8_epi16(r) };

      let rg = unsafe { _mm256_sub_epi16(r16, g16) };
      let rpg = unsafe { _mm256_add_epi16(r16, g16) };
      let two_b = unsafe { _mm256_slli_epi16::<1>(b16) };
      let u = unsafe { _mm256_sub_epi16(rpg, two_b) };

      // Σ rg / Σ u: pair-sum → i32×8, widen → i64×4 × 2, accumulate.
      let rg_pairs = unsafe { _mm256_madd_epi16(rg, ones16) };
      let rg_lo = unsafe { _mm256_castsi256_si128(rg_pairs) };
      let rg_hi = unsafe { _mm256_extracti128_si256::<1>(rg_pairs) };
      sum_rg = unsafe { _mm256_add_epi64(sum_rg, _mm256_cvtepi32_epi64(rg_lo)) };
      sum_rg = unsafe { _mm256_add_epi64(sum_rg, _mm256_cvtepi32_epi64(rg_hi)) };

      let u_pairs = unsafe { _mm256_madd_epi16(u, ones16) };
      let u_lo = unsafe { _mm256_castsi256_si128(u_pairs) };
      let u_hi = unsafe { _mm256_extracti128_si256::<1>(u_pairs) };
      sum_u = unsafe { _mm256_add_epi64(sum_u, _mm256_cvtepi32_epi64(u_lo)) };
      sum_u = unsafe { _mm256_add_epi64(sum_u, _mm256_cvtepi32_epi64(u_hi)) };

      // Σ rg² / Σ u²: madd-with-self.
      let rg_sq = unsafe { _mm256_madd_epi16(rg, rg) };
      let rg_sq_lo = unsafe { _mm256_castsi256_si128(rg_sq) };
      let rg_sq_hi = unsafe { _mm256_extracti128_si256::<1>(rg_sq) };
      sum_rg_sq = unsafe { _mm256_add_epi64(sum_rg_sq, _mm256_cvtepi32_epi64(rg_sq_lo)) };
      sum_rg_sq = unsafe { _mm256_add_epi64(sum_rg_sq, _mm256_cvtepi32_epi64(rg_sq_hi)) };

      let u_sq = unsafe { _mm256_madd_epi16(u, u) };
      let u_sq_lo = unsafe { _mm256_castsi256_si128(u_sq) };
      let u_sq_hi = unsafe { _mm256_extracti128_si256::<1>(u_sq) };
      sum_u_sq = unsafe { _mm256_add_epi64(sum_u_sq, _mm256_cvtepi32_epi64(u_sq_lo)) };
      sum_u_sq = unsafe { _mm256_add_epi64(sum_u_sq, _mm256_cvtepi32_epi64(u_sq_hi)) };

      x += LANES;
    }

    while x < w {
      let b = bgr[row_base + x * 3] as i32;
      let g = bgr[row_base + x * 3 + 1] as i32;
      let r = bgr[row_base + x * 3 + 2] as i32;
      let rg = r - g;
      let u = r + g - 2 * b;
      tail_sum_rg += rg as i64;
      tail_sum_u += u as i64;
      tail_sum_rg_sq += (rg * rg) as u64;
      tail_sum_u_sq += (u * u) as u64;
      x += 1;
    }
  }

  // Reduce i64×4 → i64.
  let reduce_signed = |v: __m256i| -> i64 {
    let mut lanes = [0i64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
    lanes.iter().sum()
  };
  let reduce_unsigned = |v: __m256i| -> u64 {
    let mut lanes = [0i64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
    lanes
      .iter()
      .fold(0u64, |acc, &x| acc.wrapping_add(x as u64))
  };

  let total_sum_rg = reduce_signed(sum_rg).wrapping_add(tail_sum_rg);
  let total_sum_u = reduce_signed(sum_u).wrapping_add(tail_sum_u);
  let total_sum_rg_sq = reduce_unsigned(sum_rg_sq).wrapping_add(tail_sum_rg_sq);
  let total_sum_u_sq = reduce_unsigned(sum_u_sq).wrapping_add(tail_sum_u_sq);

  let n_f = n as f64;
  let mean_rg = (total_sum_rg as f64) / n_f;
  let mean_u = (total_sum_u as f64) / n_f;
  let mean_yb = mean_u * 0.5;
  let var_rg = ((total_sum_rg_sq as f64) / n_f - mean_rg * mean_rg).max(0.0);
  let var_u = ((total_sum_u_sq as f64) / n_f - mean_u * mean_u).max(0.0);
  let var_yb = var_u * 0.25;

  let sigma_rgyb = crate::sqrt_64(var_rg + var_yb);
  let mu_rgyb = crate::sqrt_64(mean_rg * mean_rg + mean_yb * mean_yb);
  (sigma_rgyb + 0.3 * mu_rgyb) as f32
}

/// AVX2 magnitude-weighted gradient-direction anisotropy.
///
/// 8-pixel chunks — doubles the SSSE3 backend's 4-lane width.
/// Per chunk:
/// - Load 8 `i32` mag values via `_mm256_loadu_si256` (32
///   bytes).
/// - `_mm_loadl_epi64` reads 8 dir bytes into the low half of a
///   `__m128i`, then `_mm256_cvtepu8_epi32` zero-extends them
///   into `i32×8` lanes. AND with `3` keeps only the bin index.
/// - `_mm256_cmpgt_epi32(mag, 0)` produces the `mag > 0` mask;
///   AND with mag zeros non-positive lanes.
/// - For each bin `b ∈ {0,1,2,3}`,
///   `_mm256_cmpeq_epi32(bins, b)` yields a per-lane bin mask.
///   AND with the gated mag values; split via
///   `_mm256_castsi256_si128` / `_mm256_extracti128_si256` and
///   widen each i32×4 half to `i64×4` via
///   `_mm256_cvtepi32_epi64` (sign-extend; values are
///   non-negative). Accumulate into the bin's `i64×4`
///   accumulator.
///
/// Like the other backends, the scalar tail matches the
/// reference loop exactly and the dispatcher panics if
/// `mag.len()` or `dir.len()` is less than `w * h` before this
/// function ever runs.
///
/// # Safety
///
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[allow(unused_unsafe)]
pub(super) unsafe fn gradient_anisotropy(mag: &[i32], dir: &[u8], w: usize, h: usize) -> f32 {
  if w < 3 || h < 3 {
    return 0.0;
  }

  const LANES: usize = 8;
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  let zero = unsafe { _mm256_setzero_si256() };
  let mask3 = unsafe { _mm256_set1_epi32(0b11) };
  let bin_consts = [
    unsafe { _mm256_setzero_si256() },
    unsafe { _mm256_set1_epi32(1) },
    unsafe { _mm256_set1_epi32(2) },
    unsafe { _mm256_set1_epi32(3) },
  ];

  let mut acc: [__m256i; 4] = [unsafe { _mm256_setzero_si256() }; 4];
  let mut tail: [u64; 4] = [0; 4];

  for y in 1..h - 1 {
    let row_off = y * w;

    let mut x = 1;
    while x < x_vec_end {
      let idx = row_off + x;
      let mag8 = unsafe { _mm256_loadu_si256(mag.as_ptr().add(idx) as *const __m256i) };

      // Load 8 dir bytes into the low half of an __m128i, then
      // widen to i32×8.
      let dir8 = unsafe { _mm_loadl_epi64(dir.as_ptr().add(idx) as *const __m128i) };
      let dir_i32 = unsafe { _mm256_cvtepu8_epi32(dir8) };
      let bins_v = unsafe { _mm256_and_si256(dir_i32, mask3) };

      let pos_mask = unsafe { _mm256_cmpgt_epi32(mag8, zero) };
      let pos_mag = unsafe { _mm256_and_si256(mag8, pos_mask) };

      for bin_val in 0..4usize {
        let bin_eq = unsafe { _mm256_cmpeq_epi32(bins_v, bin_consts[bin_val]) };
        let masked = unsafe { _mm256_and_si256(pos_mag, bin_eq) };
        let lo_128 = unsafe { _mm256_castsi256_si128(masked) };
        let hi_128 = unsafe { _mm256_extracti128_si256::<1>(masked) };
        let lo64 = unsafe { _mm256_cvtepi32_epi64(lo_128) };
        let hi64 = unsafe { _mm256_cvtepi32_epi64(hi_128) };
        acc[bin_val] = unsafe { _mm256_add_epi64(acc[bin_val], lo64) };
        acc[bin_val] = unsafe { _mm256_add_epi64(acc[bin_val], hi64) };
      }

      x += LANES;
    }

    while x < w - 1 {
      let idx = row_off + x;
      let m = mag[idx];
      if m > 0 {
        let d = dir[idx] as usize & 0b11;
        tail[d] = tail[d].saturating_add(m as u64);
      }
      x += 1;
    }
  }

  let mut hist = tail;
  for bin_val in 0..4 {
    let mut lanes = [0i64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc[bin_val]) };
    let bin_sum = lanes.iter().fold(0u64, |a, &x| a.wrapping_add(x as u64));
    hist[bin_val] = hist[bin_val].wrapping_add(bin_sum);
  }

  let total: u64 = hist.iter().sum();
  if total == 0 {
    return 0.0;
  }
  let max_bin = *hist.iter().max().expect("4 bins") as f64;
  let total_f = total as f64;
  let frac = max_bin / total_f;
  ((frac - 0.25).max(0.0) / 0.75) as f32
}
