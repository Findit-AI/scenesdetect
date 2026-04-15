//! Criterion benchmark for [`Detector::process`] across typical video frame
//! sizes. Measures the full per-frame cost: area-weighted resize + DCT +
//! low-frequency crop + median + bit packing + Hamming distance +
//! bookkeeping.
//!
//! The first iteration of each bench function triggers a one-time
//! [`ResizeTable`] build for the new source resolution; criterion's
//! warmup absorbs this so reported numbers reflect steady-state cost.
//!
//! Run with `cargo bench --bench phash`.

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use scenesdetect::frame::{LumaFrame, Timebase, Timestamp};
use scenesdetect::phash::{Detector, Options};

/// Generates a deterministic pseudo-random Y-plane of the requested size.
/// Uses a tiny LCG so regenerating per benchmark group is negligible.
fn make_luma(width: u32, height: u32) -> Vec<u8> {
  let mut state: u32 = 0x9E3779B9;
  let n = (width as usize) * (height as usize);
  let mut buf = Vec::with_capacity(n);
  for _ in 0..n {
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    buf.push((state >> 24) as u8);
  }
  buf
}

fn bench_process(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("phash::Detector::process");

  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_luma(w, h);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      // Fresh detector and a frame counter so each iteration presents a
      // distinct timestamp — keeps the min_duration gate realistic.
      let mut det = Detector::new(Options::default());
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = LumaFrame::new(&buf, w, h, w, Timestamp::new(pts, tb));
        pts += 33; // ≈30 fps in 1/1000 timebase
        black_box(det.process(frame));
      });
    });
  }

  group.finish();
}

criterion_group!(benches, bench_process);
criterion_main!(benches);
