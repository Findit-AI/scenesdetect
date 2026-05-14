//! Aarch64 NEON backend for BGR→HSV (3-channel deinterleave via `vld3q_u8`).

use core::arch::aarch64::*;

#[target_feature(enable = "neon")]
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

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      // Deinterleave 16 BGR pixels (48 bytes) into three u8x16 vectors.
      let bgr = unsafe { vld3q_u8(src.as_ptr().add(row_base + x * 3)) };
      let b = bgr.0;
      let g = bgr.1;
      let r = bgr.2;

      // Per channel: u8x16 → two u16x8 halves.
      let b_lo16 = unsafe { vmovl_u8(vget_low_u8(b)) };
      let b_hi16 = unsafe { vmovl_high_u8(b) };
      let g_lo16 = unsafe { vmovl_u8(vget_low_u8(g)) };
      let g_hi16 = unsafe { vmovl_high_u8(g) };
      let r_lo16 = unsafe { vmovl_u8(vget_low_u8(r)) };
      let r_hi16 = unsafe { vmovl_high_u8(r) };

      // Four 4-pixel groups: {0..4, 4..8, 8..12, 12..16}.
      macro_rules! process_group {
        ($b16:expr, $g16:expr, $r16:expr, $half:ident) => {{
          let bu32 = unsafe { $half($b16) };
          let gu32 = unsafe { $half($g16) };
          let ru32 = unsafe { $half($r16) };
          let bf = unsafe { vcvtq_f32_u32(bu32) };
          let gf = unsafe { vcvtq_f32_u32(gu32) };
          let rf = unsafe { vcvtq_f32_u32(ru32) };
          let (hue, sat, val) = unsafe { bgr_to_hsv_f32x4(bf, gf, rf) };
          // Hue/2 → u32, clamp [0, 179]; S/V → u32, clamp [0, 255].
          let hue_half = unsafe { vmulq_n_f32(hue, 0.5) };
          let h_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(hue_half), vdupq_n_u32(179)) };
          let s_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(sat), vdupq_n_u32(255)) };
          let v_u32 = unsafe { vminq_u32(vcvtaq_u32_f32(val), vdupq_n_u32(255)) };
          (h_u32, s_u32, v_u32)
        }};
      }

      let g0 = process_group!(b_lo16, g_lo16, r_lo16, vmovl_u16_low);
      let g1 = process_group!(b_lo16, g_lo16, r_lo16, vmovl_u16_high);
      let g2 = process_group!(b_hi16, g_hi16, r_hi16, vmovl_u16_low);
      let g3 = process_group!(b_hi16, g_hi16, r_hi16, vmovl_u16_high);

      let h_bufs: [uint32x4_t; 4] = [g0.0, g1.0, g2.0, g3.0];
      let s_bufs: [uint32x4_t; 4] = [g0.1, g1.1, g2.1, g3.1];
      let v_bufs: [uint32x4_t; 4] = [g0.2, g1.2, g2.2, g3.2];

      let h_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&h_bufs) };
      let s_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&s_bufs) };
      let v_u8x16 = unsafe { pack_u32x4_quad_to_u8x16(&v_bufs) };
      unsafe {
        vst1q_u8(h_out.as_mut_ptr().add(dst_off + x), h_u8x16);
        vst1q_u8(s_out.as_mut_ptr().add(dst_off + x), s_u8x16);
        vst1q_u8(v_out.as_mut_ptr().add(dst_off + x), v_u8x16);
      }

      x += LANES;
    }

    // Scalar tail.
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

/// Widen the low four lanes of a `uint16x8_t` to `uint32x4_t`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn vmovl_u16_low(v: uint16x8_t) -> uint32x4_t {
  unsafe { vmovl_u16(vget_low_u16(v)) }
}

/// Widen the high four lanes of a `uint16x8_t` to `uint32x4_t`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn vmovl_u16_high(v: uint16x8_t) -> uint32x4_t {
  unsafe { vmovl_high_u16(v) }
}

/// Four `u32x4` → one `u8x16`, via saturating narrow. Lane order is
/// preserved: `[q[0][0..4], q[1][0..4], q[2][0..4], q[3][0..4]]`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn pack_u32x4_quad_to_u8x16(quads: &[uint32x4_t; 4]) -> uint8x16_t {
  let u16_0 = unsafe { vqmovn_u32(quads[0]) };
  let u16_1 = unsafe { vqmovn_u32(quads[1]) };
  let u16_2 = unsafe { vqmovn_u32(quads[2]) };
  let u16_3 = unsafe { vqmovn_u32(quads[3]) };
  let u16_lo = unsafe { vcombine_u16(u16_0, u16_1) };
  let u16_hi = unsafe { vcombine_u16(u16_2, u16_3) };
  let u8_lo = unsafe { vqmovn_u16(u16_lo) };
  let u8_hi = unsafe { vqmovn_u16(u16_hi) };
  unsafe { vcombine_u8(u8_lo, u8_hi) }
}

/// Branch-free 4-lane BGR→HSV core. Returns `(hue ∈ [0, 360),
/// sat ∈ [0, 255], val ∈ [0, 255])` as `f32x4`.
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
#[inline]
unsafe fn bgr_to_hsv_f32x4(
  b: float32x4_t,
  g: float32x4_t,
  r: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  let zero = unsafe { vdupq_n_f32(0.0) };
  let one = unsafe { vdupq_n_f32(1.0) };

  let v = unsafe { vmaxq_f32(vmaxq_f32(b, g), r) };
  let min = unsafe { vminq_f32(vminq_f32(b, g), r) };
  let delta = unsafe { vsubq_f32(v, min) };

  let delta_zero = unsafe { vceqq_f32(delta, zero) };
  let v_zero = unsafe { vceqq_f32(v, zero) };
  let delta_safe = unsafe { vbslq_f32(delta_zero, one, delta) };

  let sixty = unsafe { vdupq_n_f32(60.0) };
  let c120 = unsafe { vdupq_n_f32(120.0) };
  let c240 = unsafe { vdupq_n_f32(240.0) };
  let c360 = unsafe { vdupq_n_f32(360.0) };
  let c255 = unsafe { vdupq_n_f32(255.0) };

  let h_r = unsafe { vdivq_f32(vmulq_f32(sixty, vsubq_f32(g, b)), delta_safe) };
  let h_g = unsafe {
    vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(b, r)), delta_safe),
      c120,
    )
  };
  let h_b = unsafe {
    vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(r, g)), delta_safe),
      c240,
    )
  };

  let is_r = unsafe { vceqq_f32(v, r) };
  let is_g = unsafe { vceqq_f32(v, g) };
  let not_r_and_g = unsafe { vandq_u32(vmvnq_u32(is_r), is_g) };
  let hue_rg = unsafe { vbslq_f32(is_r, h_r, h_b) };
  let hue = unsafe { vbslq_f32(not_r_and_g, h_g, hue_rg) };
  let neg = unsafe { vcltq_f32(hue, zero) };
  let hue = unsafe { vbslq_f32(neg, vaddq_f32(hue, c360), hue) };
  let hue = unsafe { vbslq_f32(delta_zero, zero, hue) };

  let v_safe = unsafe { vbslq_f32(v_zero, one, v) };
  let sat = unsafe { vdivq_f32(vmulq_f32(c255, delta), v_safe) };
  let sat = unsafe { vbslq_f32(v_zero, zero, sat) };

  (hue, sat, v)
}

/// NEON `mean_abs_diff`: `Σ|a[i] - b[i]| / n`.
///
/// Uses `vabdq_u8` (absolute-difference, 16 bytes) → `vpaddlq_u8` (pairwise
/// add-long u8→u16) → `vpaddlq_u16` (u16→u32) → `vpaddlq_u32` (u32→u64),
/// accumulating into a `u64x2`. Tail handled scalar.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn mean_abs_diff(a: &[u8], b: &[u8], n: usize) -> f64 {
  const LANES: usize = 16;
  let whole = n / LANES * LANES;
  let mut acc = unsafe { vdupq_n_u64(0) }; // u64x2 accumulator

  let mut i = 0;
  while i < whole {
    let va = unsafe { vld1q_u8(a.as_ptr().add(i)) };
    let vb = unsafe { vld1q_u8(b.as_ptr().add(i)) };
    // |a - b| as u8x16.
    let diff = unsafe { vabdq_u8(va, vb) };
    // Widen + reduce: u8x16 → u16x8 → u32x4 → u64x2, each step pairwise-sums.
    let s16 = unsafe { vpaddlq_u8(diff) };
    let s32 = unsafe { vpaddlq_u16(s16) };
    let s64 = unsafe { vpaddlq_u32(s32) };
    acc = unsafe { vaddq_u64(acc, s64) };
    i += LANES;
  }

  // Horizontal reduce u64x2 → u64.
  let mut sum: u64 = unsafe { vgetq_lane_u64::<0>(acc) + vgetq_lane_u64::<1>(acc) };

  // Scalar tail.
  while i < n {
    let da = a[i] as i32 - b[i] as i32;
    sum += da.unsigned_abs() as u64;
    i += 1;
  }

  sum as f64 / n as f64
}

/// NEON Sobel 3×3. Computes Gx, Gy, magnitude in i16x8 (8 pixels/iter)
/// via shifted row loads. Direction quantization is scalar from extracted lanes.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
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

    // SIMD body: 8 pixels per iteration.
    while x + LANES < w {
      // 9 shifted loads, widen u8x8 → i16x8.
      macro_rules! ld {
        ($row:expr, $o:expr) => {{ unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8($row.as_ptr().add($o)))) } }};
      }
      let pl = ld!(prev, x - 1);
      let pm = ld!(prev, x);
      let pr = ld!(prev, x + 1);
      let cl = ld!(curr, x - 1);
      let cr = ld!(curr, x + 1);
      let nl = ld!(next, x - 1);
      let nm = ld!(next, x);
      let nr = ld!(next, x + 1);

      // Gx = (pr + 2*cr + nr) - (pl + 2*cl + nl)
      let gx = unsafe {
        let pos = vaddq_s16(vaddq_s16(pr, vshlq_n_s16::<1>(cr)), nr);
        let neg = vaddq_s16(vaddq_s16(pl, vshlq_n_s16::<1>(cl)), nl);
        vsubq_s16(pos, neg)
      };

      // Gy = (nl + 2*nm + nr) - (pl + 2*pm + pr)
      let gy = unsafe {
        let pos = vaddq_s16(vaddq_s16(nl, vshlq_n_s16::<1>(nm)), nr);
        let neg = vaddq_s16(vaddq_s16(pl, vshlq_n_s16::<1>(pm)), pr);
        vsubq_s16(pos, neg)
      };

      // mag = |gx| + |gy| as i16, then widen to i32 and store.
      let mag_i16 = unsafe { vaddq_s16(vabsq_s16(gx), vabsq_s16(gy)) };
      unsafe {
        vst1q_s32(
          mag.as_mut_ptr().add(off + x),
          vmovl_s16(vget_low_s16(mag_i16)),
        );
        vst1q_s32(mag.as_mut_ptr().add(off + x + 4), vmovl_high_s16(mag_i16));
      }

      // Direction: extract to scalar for the branchy quantization.
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
      let ax = gx.unsigned_abs();
      let ay = gy.unsigned_abs();
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

/// NEON BGR → BT.601 luma: `Y = (77·R + 150·G + 29·B) >> 8`.
/// Processes 16 pixels per iteration via `vld3q_u8` deinterleave + two
/// u8-to-u16 multiply-accumulate chains (one per 8-lane half), then
/// right-shift-narrow back to u8. Tail handled scalar.
///
/// Coefficients sum to exactly 256, so the u16 accumulator stays in
/// `[0, 65280]` with no saturation risk.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn bgr_to_luma(out: &mut [u8], src: &[u8], width: u32, height: u32, stride: u32) {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  // Splat the BT.601 coefficients once; same values for every row.
  let k_b = unsafe { vdup_n_u8(29) };
  let k_g = unsafe { vdup_n_u8(150) };
  let k_r = unsafe { vdup_n_u8(77) };

  for y in 0..h {
    let row_base = y * s;
    let dst_off = y * w;

    let mut x = 0;
    while x < whole {
      // Load and deinterleave 16 packed BGR pixels.
      let bgr = unsafe { vld3q_u8(src.as_ptr().add(row_base + x * 3)) };
      let b = bgr.0;
      let g = bgr.1;
      let r = bgr.2;

      // Low 8 lanes: acc_lo = 29·B + 150·G + 77·R, widening u8×u8→u16.
      let mut acc_lo = unsafe { vmull_u8(vget_low_u8(b), k_b) };
      acc_lo = unsafe { vmlal_u8(acc_lo, vget_low_u8(g), k_g) };
      acc_lo = unsafe { vmlal_u8(acc_lo, vget_low_u8(r), k_r) };

      // High 8 lanes.
      let mut acc_hi = unsafe { vmull_u8(vget_high_u8(b), k_b) };
      acc_hi = unsafe { vmlal_u8(acc_hi, vget_high_u8(g), k_g) };
      acc_hi = unsafe { vmlal_u8(acc_hi, vget_high_u8(r), k_r) };

      // Shift right by 8 (divide by 256) and narrow to u8.
      let y_lo = unsafe { vshrn_n_u16::<8>(acc_lo) };
      let y_hi = unsafe { vshrn_n_u16::<8>(acc_hi) };

      // Combine halves and store.
      let y_u8 = unsafe { vcombine_u8(y_lo, y_hi) };
      unsafe {
        vst1q_u8(out.as_mut_ptr().add(dst_off + x), y_u8);
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

/// NEON clipping-pixel count. 16 pixels per iteration via `vld3q_u8`
/// deinterleave; per-pixel `max(B, G, R)` via two `vmaxq_u8` calls;
/// two lane-wise compares (`vcltq_u8` and `vcgtq_u8`) OR'd together to
/// form a 0/0xFF mask per pixel. The mask is right-shifted by 7 to
/// 0/1 and horizontally reduced with `vaddvq_u8` (max sum per
/// iteration is 16, fits comfortably in u8). Tail handled scalar.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn clipping_count(src: &[u8], width: u32, height: u32, stride: u32) -> u64 {
  const LANES: usize = 16;
  let w = width as usize;
  let h = height as usize;
  let s = stride as usize;
  let whole = w / LANES * LANES;

  // Thresholds splatted once.
  let lo = unsafe { vdupq_n_u8(5) };
  let hi = unsafe { vdupq_n_u8(250) };

  let mut count: u64 = 0;
  for y in 0..h {
    let row_base = y * s;

    let mut x = 0;
    while x < whole {
      let bgr = unsafe { vld3q_u8(src.as_ptr().add(row_base + x * 3)) };
      let max = unsafe { vmaxq_u8(bgr.0, vmaxq_u8(bgr.1, bgr.2)) };
      let is_low = unsafe { vcltq_u8(max, lo) };
      let is_high = unsafe { vcgtq_u8(max, hi) };
      let mask = unsafe { vorrq_u8(is_low, is_high) };
      // Lane is either 0 or 0xFF → shift to 0 or 1, then sum.
      let zeroed = unsafe { vshrq_n_u8::<7>(mask) };
      count += unsafe { vaddvq_u8(zeroed) } as u64;
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

/// NEON Tenengrad: processes 8 interior pixels per iteration.
///
/// For each output pixel `(y, x)`, the 3×3 Sobel needs samples at
/// offsets `{-1, 0, +1}` in both axes. We load three `u8x8` vectors
/// per row (at offsets `x-1`, `x`, `x+1`) for the prev/curr/next
/// rows, widen to `i16x8`, compute `gx` / `gy` lane-wise, square via
/// `vmull_s16` / `vmull_high_s16` (i16×i16 → i32 with widening), sum
/// per-pixel and pair-add into a `i64x2` accumulator. Tail pixels
/// handled scalar.
///
/// Max per-pixel |gx| / |gy| on 8-bit input is 4·255 = 1020, well
/// within i16 range; the squared sum per pixel fits in i32; the
/// accumulator lives in i64.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
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

  // Process interior x in [1, w-1); the main loop strides LANES
  // starting at x=1, so the last vector iteration starts at
  // `1 + k*LANES` with `1 + k*LANES + LANES <= w-1`, i.e.
  // `k*LANES <= w - 2 - LANES`.
  let x_vec_end = if w >= 2 + LANES {
    1 + ((w - 2) / LANES) * LANES
  } else {
    1
  };

  // i64x2 accumulator.
  let mut acc = unsafe { vdupq_n_s64(0) };

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      // Load three u8x8 vectors per row, at offsets x-1, x, x+1.
      let prev_m1 = unsafe { vld1_u8(prev.as_ptr().add(x - 1)) };
      let prev_0 = unsafe { vld1_u8(prev.as_ptr().add(x)) };
      let prev_p1 = unsafe { vld1_u8(prev.as_ptr().add(x + 1)) };
      let curr_m1 = unsafe { vld1_u8(curr.as_ptr().add(x - 1)) };
      let curr_p1 = unsafe { vld1_u8(curr.as_ptr().add(x + 1)) };
      let next_m1 = unsafe { vld1_u8(next.as_ptr().add(x - 1)) };
      let next_0 = unsafe { vld1_u8(next.as_ptr().add(x)) };
      let next_p1 = unsafe { vld1_u8(next.as_ptr().add(x + 1)) };

      // Widen to i16x8 (zero-extend u8 then reinterpret as signed).
      let tl = unsafe { vreinterpretq_s16_u16(vmovl_u8(prev_m1)) };
      let t = unsafe { vreinterpretq_s16_u16(vmovl_u8(prev_0)) };
      let tr = unsafe { vreinterpretq_s16_u16(vmovl_u8(prev_p1)) };
      let l = unsafe { vreinterpretq_s16_u16(vmovl_u8(curr_m1)) };
      let r = unsafe { vreinterpretq_s16_u16(vmovl_u8(curr_p1)) };
      let bl = unsafe { vreinterpretq_s16_u16(vmovl_u8(next_m1)) };
      let b = unsafe { vreinterpretq_s16_u16(vmovl_u8(next_0)) };
      let br = unsafe { vreinterpretq_s16_u16(vmovl_u8(next_p1)) };

      // gx = -tl - 2·l - bl + tr + 2·r + br
      let two_l = unsafe { vshlq_n_s16::<1>(l) };
      let two_r = unsafe { vshlq_n_s16::<1>(r) };
      let pos_x = unsafe { vaddq_s16(vaddq_s16(tr, two_r), br) };
      let neg_x = unsafe { vaddq_s16(vaddq_s16(tl, two_l), bl) };
      let gx = unsafe { vsubq_s16(pos_x, neg_x) };

      // gy = -tl - 2·t - tr + bl + 2·b + br
      let two_t = unsafe { vshlq_n_s16::<1>(t) };
      let two_b = unsafe { vshlq_n_s16::<1>(b) };
      let pos_y = unsafe { vaddq_s16(vaddq_s16(bl, two_b), br) };
      let neg_y = unsafe { vaddq_s16(vaddq_s16(tl, two_t), tr) };
      let gy = unsafe { vsubq_s16(pos_y, neg_y) };

      // gx² + gy² per lane, widened to i32x4 low and i32x4 high.
      let gx_lo = unsafe { vget_low_s16(gx) };
      let gy_lo = unsafe { vget_low_s16(gy) };
      let gx_hi_half = unsafe { vget_high_s16(gx) };
      let gy_hi_half = unsafe { vget_high_s16(gy) };

      let sq_lo = unsafe { vaddq_s32(vmull_s16(gx_lo, gx_lo), vmull_s16(gy_lo, gy_lo)) };
      let sq_hi = unsafe {
        vaddq_s32(
          vmull_s16(gx_hi_half, gx_hi_half),
          vmull_s16(gy_hi_half, gy_hi_half),
        )
      };

      // Pair-add the i32x4 vectors into the i64x2 accumulator.
      acc = unsafe { vpadalq_s32(acc, sq_lo) };
      acc = unsafe { vpadalq_s32(acc, sq_hi) };

      x += LANES;
    }

    // Scalar tail for the row.
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
      let sq = (gx * gx + gy * gy) as i64;
      // Fold the tail into the same i64x2 accumulator by adding to
      // lane 0.
      let tail = unsafe { vsetq_lane_s64::<0>(sq, vdupq_n_s64(0)) };
      acc = unsafe { vaddq_s64(acc, tail) };
      x += 1;
    }
  }

  // Horizontal reduce i64x2 → i64.
  let lo = unsafe { vgetq_lane_s64::<0>(acc) };
  let hi = unsafe { vgetq_lane_s64::<1>(acc) };
  let total = lo + hi;
  ((total as f64) / (interior as f64)) as f32
}

/// NEON Immerkaer noise estimator on a u8 luma plane.
///
/// Mirrors [`tenengrad`]'s row-load + i64x2 accumulator
/// scaffolding, swapping the Sobel-squared kernel for the
/// Laplacian-of-difference `[1,-2,1;-2,4,-2;1,-2,1]` and
/// `vabsq_s16` for the squared-sum.
///
/// Per 8-pixel chunk:
/// - Load 9 `u8×8` neighborhoods (`tl…br`) via `vld1_u8`.
/// - Widen each to `s16×8` (`vmovl_u8` + reinterpret).
/// - Compute `lap = 4·c - 2·(t+b+l+r) + (tl+tr+bl+br)`
///   lanewise. Peak magnitude `16·255 = 4080` fits well inside
///   `i16`.
/// - `vabsq_s16` gives the per-pixel absolute value.
/// - `vpaddlq_s16` pair-sums into `s32×4` (max `2·4080 = 8160`).
/// - `vpadalq_s32` folds the `s32×4` into the `i64×2`
///   accumulator with pair-adds, so the eight per-chunk
///   absolutes land in the two accumulator lanes.
///
/// At the end, horizontal-reduce `i64×2 → i64` and scale by
/// `√(π/2)/6 / N_inner` to recover σₙ.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn noise(luma: &[u8], w: usize, h: usize, s: usize) -> f32 {
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

  let mut acc = unsafe { vdupq_n_s64(0) };

  for y in 1..h - 1 {
    let prev = &luma[(y - 1) * s..];
    let curr = &luma[y * s..];
    let next = &luma[(y + 1) * s..];

    let mut x = 1;
    while x < x_vec_end {
      let tl_u8 = unsafe { vld1_u8(prev.as_ptr().add(x - 1)) };
      let t_u8 = unsafe { vld1_u8(prev.as_ptr().add(x)) };
      let tr_u8 = unsafe { vld1_u8(prev.as_ptr().add(x + 1)) };
      let l_u8 = unsafe { vld1_u8(curr.as_ptr().add(x - 1)) };
      let c_u8 = unsafe { vld1_u8(curr.as_ptr().add(x)) };
      let r_u8 = unsafe { vld1_u8(curr.as_ptr().add(x + 1)) };
      let bl_u8 = unsafe { vld1_u8(next.as_ptr().add(x - 1)) };
      let b_u8 = unsafe { vld1_u8(next.as_ptr().add(x)) };
      let br_u8 = unsafe { vld1_u8(next.as_ptr().add(x + 1)) };

      let tl = unsafe { vreinterpretq_s16_u16(vmovl_u8(tl_u8)) };
      let t = unsafe { vreinterpretq_s16_u16(vmovl_u8(t_u8)) };
      let tr = unsafe { vreinterpretq_s16_u16(vmovl_u8(tr_u8)) };
      let l = unsafe { vreinterpretq_s16_u16(vmovl_u8(l_u8)) };
      let c = unsafe { vreinterpretq_s16_u16(vmovl_u8(c_u8)) };
      let r = unsafe { vreinterpretq_s16_u16(vmovl_u8(r_u8)) };
      let bl = unsafe { vreinterpretq_s16_u16(vmovl_u8(bl_u8)) };
      let b = unsafe { vreinterpretq_s16_u16(vmovl_u8(b_u8)) };
      let br = unsafe { vreinterpretq_s16_u16(vmovl_u8(br_u8)) };

      // lap = 4c - 2(t + b + l + r) + (tl + tr + bl + br)
      let four_c = unsafe { vshlq_n_s16::<2>(c) };
      let tblr = unsafe { vaddq_s16(vaddq_s16(t, b), vaddq_s16(l, r)) };
      let two_tblr = unsafe { vshlq_n_s16::<1>(tblr) };
      let corners = unsafe { vaddq_s16(vaddq_s16(tl, tr), vaddq_s16(bl, br)) };
      let lap = unsafe { vaddq_s16(vsubq_s16(four_c, two_tblr), corners) };
      let abs_lap = unsafe { vabsq_s16(lap) };

      // Pair-sum s16×8 → s32×4, then pair-add into the i64×2
      // accumulator.
      let pair = unsafe { vpaddlq_s16(abs_lap) };
      acc = unsafe { vpadalq_s32(acc, pair) };

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
      let abs = lap.unsigned_abs() as i64;
      let tail = unsafe { vsetq_lane_s64::<0>(abs, vdupq_n_s64(0)) };
      acc = unsafe { vaddq_s64(acc, tail) };
      x += 1;
    }
  }

  let lo = unsafe { vgetq_lane_s64::<0>(acc) };
  let hi = unsafe { vgetq_lane_s64::<1>(acc) };
  let total = lo + hi;

  // σₙ ≈ √(π/2) / 6 · (Σ|lap| / interior).
  const COEFF: f64 = 0.208_898_754_886_372_3;
  ((total as f64) * COEFF / (interior as f64)) as f32
}

/// NEON Hasler-Süßstrunk colourfulness on packed 24-bit BGR.
///
/// Same integer two-pass formulation as the SSSE3 backend
/// (sum / sum-of-squares per `rg = R-G` and `u = R+G-2B`, with
/// `yb = u/2` recovered post-reduction). NEON's `vld3q_u8`
/// deinterleaves 48 bytes directly into `(B, G, R)` u8×16
/// vectors — no shuffle table needed.
///
/// Per 16-pixel chunk:
/// - `vld3q_u8` → `b`, `g`, `r` as `u8×16`.
/// - Widen low / high halves of each channel to `i16×8` via
///   `vmovl_u8` / `vmovl_high_u8` + reinterpret to signed.
/// - Per half: `rg = R-G`, `u = R+G - 2B`.
/// - `vpaddlq_s16` pair-sums `i16×8 → i32×4` for `Σ rg` / `Σ u`.
/// - For `Σ rg²` / `Σ u²`: square the low / high i16×4 sub-halves
///   via `vmull_s16` / `vmull_high_s16` (i32×4 each), then
///   pair-add via `vpaddq_s32` to mirror SSSE3's
///   `_mm_madd_epi16(v, v)` semantics. Widen i32×4 → i64×2 × 2
///   and accumulate.
///
/// At the end, `vaddvq_s32` / `vaddvq_s64` reduce each
/// accumulator to a scalar; one f64 pass derives the final
/// `σ_rgyb + 0.3·μ_rgyb`.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
#[allow(unused_unsafe)]
pub(super) unsafe fn colorfulness(bgr: &[u8], w: usize, h: usize, stride: usize) -> f32 {
  let n = w.saturating_mul(h);
  if n == 0 {
    return 0.0;
  }

  const LANES: usize = 16;
  let whole = w / LANES * LANES;

  // All four accumulators are i64×2 so no realistic frame size
  // can overflow the running sums — biased 8K+ frames pushed an
  // i32×4 lane past `i32::MAX` and caused mean_rg/mean_yb to
  // diverge from scalar.
  let mut sum_rg = unsafe { vdupq_n_s64(0) };
  let mut sum_u = unsafe { vdupq_n_s64(0) };
  let mut sum_rg_sq = unsafe { vdupq_n_s64(0) };
  let mut sum_u_sq = unsafe { vdupq_n_s64(0) };
  let mut tail_sum_rg: i64 = 0;
  let mut tail_sum_u: i64 = 0;
  let mut tail_sum_rg_sq: u64 = 0;
  let mut tail_sum_u_sq: u64 = 0;

  for y in 0..h {
    let row_base = y * stride;

    let mut x = 0;
    while x < whole {
      let bgr_v = unsafe { vld3q_u8(bgr.as_ptr().add(row_base + x * 3)) };
      let b = bgr_v.0;
      let g = bgr_v.1;
      let r = bgr_v.2;

      // Low halves.
      let b_lo = unsafe { vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b))) };
      let g_lo = unsafe { vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g))) };
      let r_lo = unsafe { vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r))) };
      let rg_lo = unsafe { vsubq_s16(r_lo, g_lo) };
      let rpg_lo = unsafe { vaddq_s16(r_lo, g_lo) };
      let two_b_lo = unsafe { vshlq_n_s16::<1>(b_lo) };
      let u_lo = unsafe { vsubq_s16(rpg_lo, two_b_lo) };

      // High halves.
      let b_hi = unsafe { vreinterpretq_s16_u16(vmovl_high_u8(b)) };
      let g_hi = unsafe { vreinterpretq_s16_u16(vmovl_high_u8(g)) };
      let r_hi = unsafe { vreinterpretq_s16_u16(vmovl_high_u8(r)) };
      let rg_hi = unsafe { vsubq_s16(r_hi, g_hi) };
      let rpg_hi = unsafe { vaddq_s16(r_hi, g_hi) };
      let two_b_hi = unsafe { vshlq_n_s16::<1>(b_hi) };
      let u_hi = unsafe { vsubq_s16(rpg_hi, two_b_hi) };

      // Σ rg / Σ u: pair-sum i16×8 → i32×4 (per-half), add the
      // two halves, then sign-extend i32×4 → i64×2 × 2 before
      // accumulating so any frame size stays well inside the
      // i64×2 accumulators.
      let rg_pairs_lo = unsafe { vpaddlq_s16(rg_lo) };
      let rg_pairs_hi = unsafe { vpaddlq_s16(rg_hi) };
      let rg_pairs = unsafe { vaddq_s32(rg_pairs_lo, rg_pairs_hi) };
      sum_rg = unsafe { vaddq_s64(sum_rg, vmovl_s32(vget_low_s32(rg_pairs))) };
      sum_rg = unsafe { vaddq_s64(sum_rg, vmovl_high_s32(rg_pairs)) };
      let u_pairs_lo = unsafe { vpaddlq_s16(u_lo) };
      let u_pairs_hi = unsafe { vpaddlq_s16(u_hi) };
      let u_pairs = unsafe { vaddq_s32(u_pairs_lo, u_pairs_hi) };
      sum_u = unsafe { vaddq_s64(sum_u, vmovl_s32(vget_low_s32(u_pairs))) };
      sum_u = unsafe { vaddq_s64(sum_u, vmovl_high_s32(u_pairs)) };

      // Σ rg² (and same template for u²): square each i16 lane
      // via `vmull_s16` / `vmull_high_s16`, then `vpaddq_s32` to
      // pair-sum into i32×4 — matches `_mm_madd_epi16(v, v)`.
      let rg_lo_sq_l = unsafe { vmull_s16(vget_low_s16(rg_lo), vget_low_s16(rg_lo)) };
      let rg_lo_sq_h = unsafe { vmull_high_s16(rg_lo, rg_lo) };
      let rg_lo_pair_sq = unsafe { vpaddq_s32(rg_lo_sq_l, rg_lo_sq_h) };
      let rg_hi_sq_l = unsafe { vmull_s16(vget_low_s16(rg_hi), vget_low_s16(rg_hi)) };
      let rg_hi_sq_h = unsafe { vmull_high_s16(rg_hi, rg_hi) };
      let rg_hi_pair_sq = unsafe { vpaddq_s32(rg_hi_sq_l, rg_hi_sq_h) };
      let rg_sq_chunk = unsafe { vaddq_s32(rg_lo_pair_sq, rg_hi_pair_sq) };
      let rg_sq_lo64 = unsafe { vmovl_s32(vget_low_s32(rg_sq_chunk)) };
      let rg_sq_hi64 = unsafe { vmovl_high_s32(rg_sq_chunk) };
      sum_rg_sq = unsafe { vaddq_s64(sum_rg_sq, rg_sq_lo64) };
      sum_rg_sq = unsafe { vaddq_s64(sum_rg_sq, rg_sq_hi64) };

      let u_lo_sq_l = unsafe { vmull_s16(vget_low_s16(u_lo), vget_low_s16(u_lo)) };
      let u_lo_sq_h = unsafe { vmull_high_s16(u_lo, u_lo) };
      let u_lo_pair_sq = unsafe { vpaddq_s32(u_lo_sq_l, u_lo_sq_h) };
      let u_hi_sq_l = unsafe { vmull_s16(vget_low_s16(u_hi), vget_low_s16(u_hi)) };
      let u_hi_sq_h = unsafe { vmull_high_s16(u_hi, u_hi) };
      let u_hi_pair_sq = unsafe { vpaddq_s32(u_hi_sq_l, u_hi_sq_h) };
      let u_sq_chunk = unsafe { vaddq_s32(u_lo_pair_sq, u_hi_pair_sq) };
      let u_sq_lo64 = unsafe { vmovl_s32(vget_low_s32(u_sq_chunk)) };
      let u_sq_hi64 = unsafe { vmovl_high_s32(u_sq_chunk) };
      sum_u_sq = unsafe { vaddq_s64(sum_u_sq, u_sq_lo64) };
      sum_u_sq = unsafe { vaddq_s64(sum_u_sq, u_sq_hi64) };

      x += LANES;
    }

    // Scalar tail.
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

  let total_sum_rg = (unsafe { vaddvq_s64(sum_rg) }).wrapping_add(tail_sum_rg);
  let total_sum_u = (unsafe { vaddvq_s64(sum_u) }).wrapping_add(tail_sum_u);
  let total_sum_rg_sq = (unsafe { vaddvq_s64(sum_rg_sq) } as u64).wrapping_add(tail_sum_rg_sq);
  let total_sum_u_sq = (unsafe { vaddvq_s64(sum_u_sq) } as u64).wrapping_add(tail_sum_u_sq);

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

/// NEON magnitude-weighted gradient-direction anisotropy.
///
/// Builds `hist[k] = Σ mag[p] where dir[p] & 3 == k` over the
/// interior, treating `mag[p] <= 0` as contributing nothing.
/// Returns the normalized concentration
/// `((max(hist)/total) - 0.25).max(0) / 0.75`.
///
/// 8-pixel chunks. Per chunk:
/// - `vld1_u8` → 8 dir bytes; `vand_u8` masks to bin indices.
/// - Widen `u8×8 → u16×8` via `vmovl_u8`, then split to two
///   `i32×4` lane indices (`vmovl_u16(vget_low_u16(_))` and
///   `vmovl_high_u16(_)` + reinterpret-as-signed).
/// - `vld1q_s32 × 2` reads 8 i32 mag values.
/// - `vcgtq_s32(mag, 0)` produces the `mag > 0` mask. AND with
///   `mag` zeroes non-positive lanes.
/// - For each bin `b ∈ {0,1,2,3}`, `vceqq_s32(bins, b)` yields a
///   per-lane bin mask. AND with the gated mag values, widen
///   `i32×4 → i64×2 × 2` via `vmovl_s32` / `vmovl_high_s32` and
///   accumulate into the bin's i64×2 accumulator.
///
/// The 8-pixel chunk reads 8 contiguous dir bytes and 8 mag i32
/// values; bound analysis matches the scalar reference's
/// interior iteration (`x + 7 ≤ w - 2`).
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
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

  let mask3_u8 = unsafe { vdup_n_u8(0b11) };
  let zero32 = unsafe { vdupq_n_s32(0) };

  let mut acc: [int64x2_t; 4] = [unsafe { vdupq_n_s64(0) }; 4];
  let mut tail: [u64; 4] = [0; 4];

  for y in 1..h - 1 {
    let row_off = y * w;

    let mut x = 1;
    while x < x_vec_end {
      let idx = row_off + x;

      let dir8 = unsafe { vld1_u8(dir.as_ptr().add(idx)) };
      let bins8 = unsafe { vand_u8(dir8, mask3_u8) };
      let bins16 = unsafe { vmovl_u8(bins8) };
      let bins_lo_i32 = unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(bins16))) };
      let bins_hi_i32 = unsafe { vreinterpretq_s32_u32(vmovl_high_u16(bins16)) };

      let mag_lo = unsafe { vld1q_s32(mag.as_ptr().add(idx)) };
      let mag_hi = unsafe { vld1q_s32(mag.as_ptr().add(idx + 4)) };

      let pos_lo = unsafe { vcgtq_s32(mag_lo, zero32) };
      let pos_hi = unsafe { vcgtq_s32(mag_hi, zero32) };
      let pos_mag_lo = unsafe { vandq_s32(mag_lo, vreinterpretq_s32_u32(pos_lo)) };
      let pos_mag_hi = unsafe { vandq_s32(mag_hi, vreinterpretq_s32_u32(pos_hi)) };

      for b in 0..4i32 {
        let b_v = unsafe { vdupq_n_s32(b) };
        let eq_lo = unsafe { vceqq_s32(bins_lo_i32, b_v) };
        let eq_hi = unsafe { vceqq_s32(bins_hi_i32, b_v) };
        let masked_lo = unsafe { vandq_s32(pos_mag_lo, vreinterpretq_s32_u32(eq_lo)) };
        let masked_hi = unsafe { vandq_s32(pos_mag_hi, vreinterpretq_s32_u32(eq_hi)) };
        let bin_idx = b as usize;
        acc[bin_idx] = unsafe { vaddq_s64(acc[bin_idx], vmovl_s32(vget_low_s32(masked_lo))) };
        acc[bin_idx] = unsafe { vaddq_s64(acc[bin_idx], vmovl_high_s32(masked_lo)) };
        acc[bin_idx] = unsafe { vaddq_s64(acc[bin_idx], vmovl_s32(vget_low_s32(masked_hi))) };
        acc[bin_idx] = unsafe { vaddq_s64(acc[bin_idx], vmovl_high_s32(masked_hi)) };
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

  // Final scalar combine uses `saturating_add` to match the
  // scalar reference and the tail-loop `tail[d].saturating_add`
  // above. The SIMD lanes wrap internally (NEON has no
  // saturating i64 add), but the documented overflow bound
  // (`< 1.7·10¹⁶` even on 4K×2K) leaves both options
  // behaviourally identical for realistic inputs.
  let mut hist = tail;
  for bin_idx in 0..4 {
    let bin_sum = unsafe { vaddvq_s64(acc[bin_idx]) } as u64;
    hist[bin_idx] = hist[bin_idx].saturating_add(bin_sum);
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

/// NEON single-pass `(mean, variance)` on a u8 plane. Per 16-byte
/// chunk: horizontal-add-long u8×16 → u16 for `sum_x` via
/// `vaddlvq_u8`; squared sum via `vmull_u8` (low half) + `vmull_high_u8`
/// (high half) producing two u16×8 vectors whose lane max is 255² =
/// 65025, combined via `vaddlvq_u16` × 2 into u32 scalars and added
/// to a u64 accumulator.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
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
      let v = unsafe { vld1q_u8(plane.as_ptr().add(row_base + x)) };
      sum += unsafe { vaddlvq_u8(v) } as u64;

      // Per-lane u8² via widening multiply. Max per lane 255² = 65025.
      let sq_lo = unsafe { vmull_u8(vget_low_u8(v), vget_low_u8(v)) };
      let sq_hi = unsafe { vmull_high_u8(v, v) };
      sum_sq += (unsafe { vaddlvq_u16(sq_lo) } as u64) + (unsafe { vaddlvq_u16(sq_hi) } as u64);

      x += LANES;
    }

    // Scalar tail.
    while x < w {
      let v = plane[row_base + x] as u64;
      sum += v;
      sum_sq += v * v;
      x += 1;
    }
  }

  let n_f = n as f64;
  let mean = (sum as f64) / n_f;
  let mean_sq = (sum_sq as f64) / n_f;
  let variance = (mean_sq - mean * mean).max(0.0);
  (mean as f32, variance as f32)
}
