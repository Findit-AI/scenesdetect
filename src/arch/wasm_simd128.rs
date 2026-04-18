//! wasm32 SIMD128 backend for BGR→HSV.
//!
//! Same structure as the SSSE3 backend: 16 pixels per iteration,
//! `u8x16_swizzle` for 3-channel deinterleave (wasm's `swizzle` mirrors
//! x86's `PSHUFB` — mask values outside `0..16` produce zero).
//!
//! Requires the `simd128` target feature. Gated by `#[cfg(all(target_arch
//! = "wasm32", target_feature = "simd128"))]` at the dispatcher.

use core::arch::wasm32::*;

const BLK0_B: [u8; 16] = [
  0, 3, 6, 9, 12, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK0_G: [u8; 16] = [
  1, 4, 7, 10, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK0_R: [u8; 16] = [
  2, 5, 8, 11, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK1_B: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 2, 5, 8, 11, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK1_G: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 3, 6, 9, 12, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK1_R: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 4, 7, 10, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
const BLK2_B: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 4, 7, 10, 13,
];
const BLK2_G: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 2, 5, 8, 11, 14,
];
const BLK2_R: [u8; 16] = [
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 3, 6, 9, 12, 15,
];

/// wasm SIMD128 BGR→HSV: 16 pixels per iteration.
///
/// # Safety
///
/// Caller must ensure the `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
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

  let m_b0 = unsafe { v128_load(BLK0_B.as_ptr() as *const v128) };
  let m_g0 = unsafe { v128_load(BLK0_G.as_ptr() as *const v128) };
  let m_r0 = unsafe { v128_load(BLK0_R.as_ptr() as *const v128) };
  let m_b1 = unsafe { v128_load(BLK1_B.as_ptr() as *const v128) };
  let m_g1 = unsafe { v128_load(BLK1_G.as_ptr() as *const v128) };
  let m_r1 = unsafe { v128_load(BLK1_R.as_ptr() as *const v128) };
  let m_b2 = unsafe { v128_load(BLK2_B.as_ptr() as *const v128) };
  let m_g2 = unsafe { v128_load(BLK2_G.as_ptr() as *const v128) };
  let m_r2 = unsafe { v128_load(BLK2_R.as_ptr() as *const v128) };
  let zero = f32x4_splat(0.0);

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { v128_load(p as *const v128) };
      let blk1 = unsafe { v128_load(p.add(16) as *const v128) };
      let blk2 = unsafe { v128_load(p.add(32) as *const v128) };

      let b = v128_or(
        v128_or(u8x16_swizzle(blk0, m_b0), u8x16_swizzle(blk1, m_b1)),
        u8x16_swizzle(blk2, m_b2),
      );
      let g = v128_or(
        v128_or(u8x16_swizzle(blk0, m_g0), u8x16_swizzle(blk1, m_g1)),
        u8x16_swizzle(blk2, m_g2),
      );
      let r = v128_or(
        v128_or(u8x16_swizzle(blk0, m_r0), u8x16_swizzle(blk1, m_r1)),
        u8x16_swizzle(blk2, m_r2),
      );

      // Widen u8x16 → two u16x8 halves per channel.
      let b_lo16 = u16x8_extend_low_u8x16(b);
      let b_hi16 = u16x8_extend_high_u8x16(b);
      let g_lo16 = u16x8_extend_low_u8x16(g);
      let g_hi16 = u16x8_extend_high_u8x16(g);
      let r_lo16 = u16x8_extend_low_u8x16(r);
      let r_hi16 = u16x8_extend_high_u8x16(r);

      macro_rules! group {
        ($b16:expr, $g16:expr, $r16:expr, $half:ident) => {{
          let bu = $half($b16);
          let gu = $half($g16);
          let ru = $half($r16);
          let bf = f32x4_convert_u32x4(bu);
          let gf = f32x4_convert_u32x4(gu);
          let rf = f32x4_convert_u32x4(ru);
          let (hue, sat, val) = bgr_to_hsv_f32x4(bf, gf, rf);
          let hh = f32x4_mul(hue, f32x4_splat(0.5));
          let h_u32 = clamp_i32_max(i32x4_trunc_sat_f32x4(round_half(hh)), 179);
          let s_u32 = clamp_i32_max(i32x4_trunc_sat_f32x4(round_half(sat)), 255);
          let v_u32 = clamp_i32_max(i32x4_trunc_sat_f32x4(round_half(val)), 255);
          (h_u32, s_u32, v_u32)
        }};
      }

      let (h0, s0, v0) = group!(b_lo16, g_lo16, r_lo16, u32x4_extend_low_u16x8);
      let (h1, s1, v1) = group!(b_lo16, g_lo16, r_lo16, u32x4_extend_high_u16x8);
      let (h2, s2, v2) = group!(b_hi16, g_hi16, r_hi16, u32x4_extend_low_u16x8);
      let (h3, s3, v3) = group!(b_hi16, g_hi16, r_hi16, u32x4_extend_high_u16x8);

      let h_vec = pack_quad(h0, h1, h2, h3);
      let s_vec = pack_quad(s0, s1, s2, s3);
      let v_vec = pack_quad(v0, v1, v2, v3);

      unsafe {
        v128_store(h_out.as_mut_ptr().add(dst_off + x) as *mut v128, h_vec);
        v128_store(s_out.as_mut_ptr().add(dst_off + x) as *mut v128, s_vec);
        v128_store(v_out.as_mut_ptr().add(dst_off + x) as *mut v128, v_vec);
      }

      x += LANES;
    }

    // Tail.
    let _ = zero;
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

/// wasm SIMD has no direct "round away from zero"; emulate by adding 0.5
/// copysign-ed toward the input before truncating. Inputs are non-negative
/// in this pipeline so plain `+ 0.5` works.
#[target_feature(enable = "simd128")]
#[inline]
fn round_half(v: v128) -> v128 {
  f32x4_add(v, f32x4_splat(0.5))
}

/// Clamp `i32x4` lanes to `[0, max]`. Values are non-negative by construction.
#[target_feature(enable = "simd128")]
#[inline]
fn clamp_i32_max(v: v128, max: i32) -> v128 {
  let mv = i32x4_splat(max);
  let gt = i32x4_gt(v, mv);
  v128_bitselect(mv, v, gt)
}

/// Four `i32x4` (values ≤ 255) → one `u8x16` via saturating narrows.
#[target_feature(enable = "simd128")]
#[inline]
fn pack_quad(a: v128, b: v128, c: v128, d: v128) -> v128 {
  // i32x4 × 2 → i16x8 (signed saturating narrow; values 0..255 OK).
  let lo = i16x8_narrow_i32x4(a, b);
  let hi = i16x8_narrow_i32x4(c, d);
  // i16x8 × 2 → u8x16 (unsigned saturating narrow).
  u8x16_narrow_i16x8(lo, hi)
}

/// Branch-free 4-lane BGR→HSV core. Returns `(hue ∈ [0, 360), sat, val)`
/// as `f32x4`. Caller divides hue by 2 and narrows to u8.
#[target_feature(enable = "simd128")]
#[inline]
fn bgr_to_hsv_f32x4(b: v128, g: v128, r: v128) -> (v128, v128, v128) {
  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);

  let v = f32x4_max(f32x4_max(b, g), r);
  let min = f32x4_min(f32x4_min(b, g), r);
  let delta = f32x4_sub(v, min);

  let delta_zero = f32x4_eq(delta, zero);
  let v_zero = f32x4_eq(v, zero);
  // `v128_bitselect(t, f, mask)`: result = (mask & t) | (!mask & f).
  let delta_safe = v128_bitselect(one, delta, delta_zero);

  let sixty = f32x4_splat(60.0);
  let c120 = f32x4_splat(120.0);
  let c240 = f32x4_splat(240.0);
  let c360 = f32x4_splat(360.0);
  let c255 = f32x4_splat(255.0);

  let h_r = f32x4_div(f32x4_mul(sixty, f32x4_sub(g, b)), delta_safe);
  let h_g = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(b, r)), delta_safe),
    c120,
  );
  let h_b = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(r, g)), delta_safe),
    c240,
  );

  let is_r = f32x4_eq(v, r);
  let is_g = f32x4_eq(v, g);
  let not_r_and_g = v128_and(v128_not(is_r), is_g);
  let hue_rg = v128_bitselect(h_r, h_b, is_r);
  let hue = v128_bitselect(h_g, hue_rg, not_r_and_g);
  let neg = f32x4_lt(hue, zero);
  let hue = v128_bitselect(f32x4_add(hue, c360), hue, neg);
  let hue = v128_bitselect(zero, hue, delta_zero);

  let v_safe = v128_bitselect(one, v, v_zero);
  let sat = f32x4_div(f32x4_mul(c255, delta), v_safe);
  let sat = v128_bitselect(zero, sat, v_zero);

  (hue, sat, v)
}

/// wasm SIMD128 `mean_abs_diff`: `Σ|a[i] - b[i]| / n`.
///
/// Computes `|a - b|` via `max(a, b) - min(a, b)` (both saturating-safe),
/// then widens u8→u16→u32→u64 with pairwise adds for accumulation. Tail
/// handled scalar.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  const LANES: usize = 16;
  let whole = n / LANES * LANES;

  // Accumulate into two u64 lanes.
  let mut acc_lo: u64 = 0;
  let mut acc_hi: u64 = 0;

  let mut i = 0;
  while i < whole {
    let va = unsafe { v128_load(a.as_ptr().add(i) as *const v128) };
    let vb = unsafe { v128_load(b.as_ptr().add(i) as *const v128) };
    // |a - b| = max(a,b) - min(a,b) (both saturating unsigned).
    let diff = u8x16_sub_sat(u8x16_max(va, vb), u8x16_min(va, vb));
    // Widen and reduce: u8x16 → u16x8 (extend low + extend high, then add).
    let lo16 = u16x8_extend_low_u8x16(diff);
    let hi16 = u16x8_extend_high_u8x16(diff);
    let sum16 = u16x8_add(lo16, hi16); // u16x8: 8 partial sums
    // u16x8 → u32x4 → u64x2.
    let lo32 = u32x4_extend_low_u16x8(sum16);
    let hi32 = u32x4_extend_high_u16x8(sum16);
    let sum32 = u32x4_add(lo32, hi32);
    let lo64 = u64x2_extend_low_u32x4(sum32);
    let hi64 = u64x2_extend_high_u32x4(sum32);
    let sum64 = u64x2_add(lo64, hi64); // u64x2: 2 partial sums
    // Extract lanes (wasm has no u64 extract; transmute to array).
    // SAFETY: v128 and [u64; 2] have the same size and alignment.
    let arr: [u64; 2] = unsafe { core::mem::transmute(sum64) };
    acc_lo += arr[0];
    acc_hi += arr[1];
    i += LANES;
  }

  let mut sum = acc_lo + acc_hi;

  // Scalar tail.
  while i < n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
    i += 1;
  }

  sum as f64 / n as f64
}

/// wasm SIMD128 Sobel 3×3. Same structure as NEON/SSSE3: i16x8 stencil for
/// magnitude, scalar direction.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
pub(super) unsafe fn sobel(input: &[u8], mag: &mut [i32], dir: &mut [u8], w: usize, h: usize) {
  mag.fill(0);
  dir.fill(0);

  const LANES: usize = 8;

  for y in 1..h.saturating_sub(1) {
    let prev = &input[(y - 1) * w..];
    let curr = &input[y * w..];
    let next = &input[(y + 1) * w..];
    let off = y * w;

    let mut x = 1usize;

    while x + LANES <= w - 1 {
      macro_rules! ld {
        ($row:expr, $o:expr) => {{
          // Load 8 bytes, widen to i16x8.
          let v = unsafe { v128_load64_zero($row.as_ptr().add($o) as *const u64) };
          i16x8_extend_low_u8x16(v)
        }};
      }
      let pl = ld!(prev, x - 1);
      let pm = ld!(prev, x);
      let pr = ld!(prev, x + 1);
      let cl = ld!(curr, x - 1);
      let cr = ld!(curr, x + 1);
      let nl = ld!(next, x - 1);
      let nm = ld!(next, x);
      let nr = ld!(next, x + 1);

      let gx = {
        let pos = i16x8_add(i16x8_add(pr, i16x8_shl(cr, 1)), nr);
        let neg = i16x8_add(i16x8_add(pl, i16x8_shl(cl, 1)), nl);
        i16x8_sub(pos, neg)
      };
      let gy = {
        let pos = i16x8_add(i16x8_add(nl, i16x8_shl(nm, 1)), nr);
        let neg = i16x8_add(i16x8_add(pl, i16x8_shl(pm, 1)), pr);
        i16x8_sub(pos, neg)
      };

      let mag_i16 = i16x8_add(i16x8_abs(gx), i16x8_abs(gy));

      // Widen i16→i32 and store. Use signed extend.
      let mag_lo = i32x4_extend_low_i16x8(mag_i16);
      let mag_hi = i32x4_extend_high_i16x8(mag_i16);
      unsafe {
        v128_store(mag.as_mut_ptr().add(off + x) as *mut v128, mag_lo);
        v128_store(mag.as_mut_ptr().add(off + x + 4) as *mut v128, mag_hi);
      }

      // Direction: scalar.
      // SAFETY: v128 and [i16; 8] have the same size and alignment.
      let gx_arr: [i16; 8] = unsafe { core::mem::transmute(gx) };
      let gy_arr: [i16; 8] = unsafe { core::mem::transmute(gy) };
      for j in 0..LANES {
        let ax = gx_arr[j].unsigned_abs() as u32;
        let ay = gy_arr[j].unsigned_abs() as u32;
        dir[off + x + j] = if ay * 1000 < ax * 414 {
          0
        } else if ay * 1000 > ax * 2414 {
          2
        } else if (gx_arr[j] >= 0) == (gy_arr[j] >= 0) {
          1
        } else {
          3
        };
      }

      x += LANES;
    }

    // Scalar tail.
    while x < w - 1 {
      let i = |yy: usize, xx: usize| input[yy * w + xx] as i32;
      let gx = -i(y - 1, x - 1) - 2 * i(y, x - 1) - i(y + 1, x - 1)
        + i(y - 1, x + 1)
        + 2 * i(y, x + 1)
        + i(y + 1, x + 1);
      let gy = -i(y - 1, x - 1) - 2 * i(y - 1, x) - i(y - 1, x + 1)
        + i(y + 1, x - 1)
        + 2 * i(y + 1, x)
        + i(y + 1, x + 1);
      mag[off + x] = gx.abs() + gy.abs();
      let ax = gx.abs() as u32;
      let ay = gy.abs() as u32;
      dir[off + x] = if ay * 1000 < ax * 414 {
        0
      } else if ay * 1000 > ax * 2414 {
        2
      } else if gx.signum() == gy.signum() {
        1
      } else {
        3
      };
      x += 1;
    }
  }
}

/// wasm simd128 BGR → BT.601 luma: `Y = (77·R + 150·G + 29·B) >> 8`.
/// Same 9-swizzle deinterleave as [`bgr_to_hsv_planes`] → three
/// `u8x16` channel vectors; widened to `u16x8` halves via
/// `u16x8_extend_low_u8x16` / `_high_` and combined with `i16x8_mul` +
/// `i16x8_add`. Accumulator tops at 65280 in `[0, u16::MAX]` — no
/// saturation. Tail handled scalar.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_luma(out: &mut [u8], src: &[u8], width: u32, height: u32, stride: u32) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let m_b0 = unsafe { v128_load(BLK0_B.as_ptr() as *const v128) };
  let m_g0 = unsafe { v128_load(BLK0_G.as_ptr() as *const v128) };
  let m_r0 = unsafe { v128_load(BLK0_R.as_ptr() as *const v128) };
  let m_b1 = unsafe { v128_load(BLK1_B.as_ptr() as *const v128) };
  let m_g1 = unsafe { v128_load(BLK1_G.as_ptr() as *const v128) };
  let m_r1 = unsafe { v128_load(BLK1_R.as_ptr() as *const v128) };
  let m_b2 = unsafe { v128_load(BLK2_B.as_ptr() as *const v128) };
  let m_g2 = unsafe { v128_load(BLK2_G.as_ptr() as *const v128) };
  let m_r2 = unsafe { v128_load(BLK2_R.as_ptr() as *const v128) };

  // Coefficient splats. i16x8 lanes; values fit in i16.
  let k_b = i16x8_splat(29);
  let k_g = i16x8_splat(150);
  let k_r = i16x8_splat(77);

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { v128_load(p as *const v128) };
      let blk1 = unsafe { v128_load(p.add(16) as *const v128) };
      let blk2 = unsafe { v128_load(p.add(32) as *const v128) };

      let b = v128_or(
        v128_or(u8x16_swizzle(blk0, m_b0), u8x16_swizzle(blk1, m_b1)),
        u8x16_swizzle(blk2, m_b2),
      );
      let g = v128_or(
        v128_or(u8x16_swizzle(blk0, m_g0), u8x16_swizzle(blk1, m_g1)),
        u8x16_swizzle(blk2, m_g2),
      );
      let r = v128_or(
        v128_or(u8x16_swizzle(blk0, m_r0), u8x16_swizzle(blk1, m_r1)),
        u8x16_swizzle(blk2, m_r2),
      );

      // Low 8 lanes.
      let b_lo = u16x8_extend_low_u8x16(b);
      let g_lo = u16x8_extend_low_u8x16(g);
      let r_lo = u16x8_extend_low_u8x16(r);
      let acc_lo = i16x8_add(
        i16x8_add(i16x8_mul(b_lo, k_b), i16x8_mul(g_lo, k_g)),
        i16x8_mul(r_lo, k_r),
      );

      // High 8 lanes.
      let b_hi = u16x8_extend_high_u8x16(b);
      let g_hi = u16x8_extend_high_u8x16(g);
      let r_hi = u16x8_extend_high_u8x16(r);
      let acc_hi = i16x8_add(
        i16x8_add(i16x8_mul(b_hi, k_b), i16x8_mul(g_hi, k_g)),
        i16x8_mul(r_hi, k_r),
      );

      // Logical >>8 on each half, then narrow-pack to u8×16. Pack is
      // saturating-to-u8 from i16; our inputs are in [0, 255] post-
      // shift so saturation is a no-op.
      let y_lo = u16x8_shr(acc_lo, 8);
      let y_hi = u16x8_shr(acc_hi, 8);
      let packed = u8x16_narrow_i16x8(y_lo, y_hi);
      unsafe {
        v128_store(out.as_mut_ptr().add(dst_off + x) as *mut v128, packed);
      }

      x += LANES;
    }

    // Scalar tail.
    while x < w {
      let b = src[row_base + x * 3] as u32;
      let g = src[row_base + x * 3 + 1] as u32;
      let r = src[row_base + x * 3 + 2] as u32;
      out[dst_off + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
      x += 1;
    }
  }
}

/// wasm clipping-pixel count. Same 9-swizzle deinterleave as
/// [`bgr_to_hsv_planes`] produces `u8x16` B/G/R; `u8x16_max`
/// aggregates to `max(B, G, R)`; two compares (`u8x16_lt` against 5,
/// `u8x16_gt` against 250) OR'd give a 0/0xFF mask; `i8x16_bitmask`
/// returns a 16-bit popcount-ready word whose `count_ones` is the
/// lane-count per iteration. Tail scalar.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
#[allow(unused_unsafe)]
pub(super) unsafe fn clipping_count(src: &[u8], width: u32, height: u32, stride: u32) -> u64 {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  let m_b0 = unsafe { v128_load(BLK0_B.as_ptr() as *const v128) };
  let m_g0 = unsafe { v128_load(BLK0_G.as_ptr() as *const v128) };
  let m_r0 = unsafe { v128_load(BLK0_R.as_ptr() as *const v128) };
  let m_b1 = unsafe { v128_load(BLK1_B.as_ptr() as *const v128) };
  let m_g1 = unsafe { v128_load(BLK1_G.as_ptr() as *const v128) };
  let m_r1 = unsafe { v128_load(BLK1_R.as_ptr() as *const v128) };
  let m_b2 = unsafe { v128_load(BLK2_B.as_ptr() as *const v128) };
  let m_g2 = unsafe { v128_load(BLK2_G.as_ptr() as *const v128) };
  let m_r2 = unsafe { v128_load(BLK2_R.as_ptr() as *const v128) };

  let lo = u8x16_splat(5);
  let hi = u8x16_splat(250);

  let mut count: u64 = 0;
  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let p = unsafe { src.as_ptr().add(row_base + x * 3) };
      let blk0 = unsafe { v128_load(p as *const v128) };
      let blk1 = unsafe { v128_load(p.add(16) as *const v128) };
      let blk2 = unsafe { v128_load(p.add(32) as *const v128) };

      let b = v128_or(
        v128_or(u8x16_swizzle(blk0, m_b0), u8x16_swizzle(blk1, m_b1)),
        u8x16_swizzle(blk2, m_b2),
      );
      let g = v128_or(
        v128_or(u8x16_swizzle(blk0, m_g0), u8x16_swizzle(blk1, m_g1)),
        u8x16_swizzle(blk2, m_g2),
      );
      let r = v128_or(
        v128_or(u8x16_swizzle(blk0, m_r0), u8x16_swizzle(blk1, m_r1)),
        u8x16_swizzle(blk2, m_r2),
      );

      let max = u8x16_max(b, u8x16_max(g, r));
      let under = u8x16_lt(max, lo);
      let over = u8x16_gt(max, hi);
      let mask = v128_or(under, over);

      // `i8x16_bitmask` returns an i32 whose low 16 bits each reflect
      // one lane's sign bit. Since our mask lanes are 0x00 or 0xFF
      // (i.e. sign bit = 0 or 1), `count_ones` on the bitmask gives
      // the number of clipped pixels in this iteration.
      count += i8x16_bitmask(mask).count_ones() as u64;

      x += LANES;
    }

    // Scalar tail.
    while x < w {
      let b = src[row_base + x * 3];
      let g = src[row_base + x * 3 + 1];
      let r = src[row_base + x * 3 + 2];
      let m = b.max(g).max(r);
      if !(5..=250).contains(&m) {
        count += 1;
      }
      x += 1;
    }
  }

  count
}

/// wasm Tenengrad: 8 interior pixels per iteration, same structure as
/// the SSSE3 backend. Loads u8×8 chunks via `v128_load64_zero`, widens
/// to i16×8 via `u16x8_extend_low_u8x16`, computes gx/gy with
/// `i16x8_add`/`_sub`/`_shl`, interleaves gx|gy and runs the wasm
/// equivalent of PMADDWD (`i32x4_dot_i16x8`) for per-pixel
/// `gx² + gy²`, widens to i64 and accumulates.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
#[allow(unused_unsafe)]
pub(super) unsafe fn tenengrad(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
  if w < 3 || h < 3 {
    return 0.0;
  }
  let interior = (w - 2) * (h - 2);
  if interior == 0 {
    return 0.0;
  }

  const LANES: usize = 8;
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  let mut acc = i64x2_splat(0);
  let mut tail_acc: i64 = 0;

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      let load8 = |p: *const u8| -> v128 { unsafe { v128_load64_zero(p as *const u64) } };

      let tl = load8(unsafe { prev.as_ptr().add(x - 1) });
      let t = load8(unsafe { prev.as_ptr().add(x) });
      let tr = load8(unsafe { prev.as_ptr().add(x + 1) });
      let l = load8(unsafe { curr.as_ptr().add(x - 1) });
      let r = load8(unsafe { curr.as_ptr().add(x + 1) });
      let bl = load8(unsafe { next.as_ptr().add(x - 1) });
      let b = load8(unsafe { next.as_ptr().add(x) });
      let br = load8(unsafe { next.as_ptr().add(x + 1) });

      // Zero-extend u8 → i16×8.
      let tl = u16x8_extend_low_u8x16(tl);
      let t = u16x8_extend_low_u8x16(t);
      let tr = u16x8_extend_low_u8x16(tr);
      let l = u16x8_extend_low_u8x16(l);
      let r = u16x8_extend_low_u8x16(r);
      let bl = u16x8_extend_low_u8x16(bl);
      let b = u16x8_extend_low_u8x16(b);
      let br = u16x8_extend_low_u8x16(br);

      let two_l = i16x8_shl(l, 1);
      let two_r = i16x8_shl(r, 1);
      let pos_x = i16x8_add(i16x8_add(tr, two_r), br);
      let neg_x = i16x8_add(i16x8_add(tl, two_l), bl);
      let gx = i16x8_sub(pos_x, neg_x);

      let two_t = i16x8_shl(t, 1);
      let two_b = i16x8_shl(b, 1);
      let pos_y = i16x8_add(i16x8_add(bl, two_b), br);
      let neg_y = i16x8_add(i16x8_add(tl, two_t), tr);
      let gy = i16x8_sub(pos_y, neg_y);

      // `i32x4_dot_i16x8(a, b)` computes `a[i]*b[i] + a[i+1]*b[i+1]`
      // per i32 lane — the same semantic as x86's PMADDWD.
      //
      // To get `gx² + gy²` per pixel, interleave gx and gy at i16
      // granularity, then dot with itself.
      let lo_pair = u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(gx, gy);
      let hi_pair = u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(gx, gy);
      let sq_lo = i32x4_dot_i16x8(lo_pair, lo_pair); // pixels 0..4
      let sq_hi = i32x4_dot_i16x8(hi_pair, hi_pair); // pixels 4..8
      let sum32 = i32x4_add(sq_lo, sq_hi);

      // Widen i32×4 → i64×4 via sign-extend (values are non-negative
      // squared sums; any extend works).
      let sum64_a = i64x2_extend_low_i32x4(sum32);
      let sum64_b = i64x2_extend_high_i32x4(sum32);
      acc = i64x2_add(acc, sum64_a);
      acc = i64x2_add(acc, sum64_b);

      x += LANES;
    }

    // Scalar tail.
    while x < w - 1 {
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
      tail_acc += (gx * gx + gy * gy) as i64;
      x += 1;
    }
  }

  let vec_sum = i64x2_extract_lane::<0>(acc) + i64x2_extract_lane::<1>(acc);
  (((vec_sum + tail_acc) as f64) / (interior as f64)) as f32
}

/// wasm single-pass `(mean, variance)` on a u8 plane.
/// Per iter: u8 horizontal sum via `u16x8_extadd_pairwise_u8x16` ×
/// several stages into i64 lanes. Squared sum: widen u8×16 → u16×8
/// halves, per-lane `i16x8_mul` for u8², then the same pairwise-add
/// chain to u64.
///
/// # Safety
///
/// Caller must ensure `simd128` target feature is enabled.
#[target_feature(enable = "simd128")]
#[allow(unused_unsafe)]
pub(super) unsafe fn plane_mean_variance(plane: &[u8], w: usize, h: usize, s: usize) -> (f32, f32) {
  const LANES: usize = 16;
  let n = w.saturating_mul(h);
  if n == 0 {
    return (0.0, 0.0);
  }
  let whole = w / LANES * LANES;

  let mut sum: u64 = 0;
  let mut sum_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let v = unsafe { v128_load(plane.as_ptr().add(row_base + x) as *const v128) };

      // Sum: u8×16 → u16×8 (pair-add) → u32×4 (pair-add) → extract and sum.
      let s16 = u16x8_extadd_pairwise_u8x16(v);
      let s32 = u32x4_extadd_pairwise_u16x8(s16);
      sum += (u32x4_extract_lane::<0>(s32) as u64)
        + (u32x4_extract_lane::<1>(s32) as u64)
        + (u32x4_extract_lane::<2>(s32) as u64)
        + (u32x4_extract_lane::<3>(s32) as u64);

      // Squared sum: widen to u16 halves, square, then fold to u32 and extract.
      let v_lo = u16x8_extend_low_u8x16(v);
      let v_hi = u16x8_extend_high_u8x16(v);
      let sq_lo = i16x8_mul(v_lo, v_lo); // per-lane u8² ≤ 65025, fits in u16
      let sq_hi = i16x8_mul(v_hi, v_hi);
      let sq_sum_lo = u32x4_extadd_pairwise_u16x8(sq_lo);
      let sq_sum_hi = u32x4_extadd_pairwise_u16x8(sq_hi);
      let sq_sum = i32x4_add(sq_sum_lo, sq_sum_hi);
      sum_sq += (u32x4_extract_lane::<0>(sq_sum) as u64)
        + (u32x4_extract_lane::<1>(sq_sum) as u64)
        + (u32x4_extract_lane::<2>(sq_sum) as u64)
        + (u32x4_extract_lane::<3>(sq_sum) as u64);

      x += LANES;
    }

    while x < w {
      let vv = plane[row_base + x] as u64;
      sum += vv;
      sum_sq += vv * vv;
      x += 1;
    }
  }

  let n_f = n as f64;
  let mean = (sum as f64) / n_f;
  let mean_sq = (sum_sq as f64) / n_f;
  let variance = (mean_sq - mean * mean).max(0.0);
  (mean as f32, variance as f32)
}
