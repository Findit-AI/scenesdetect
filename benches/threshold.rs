//! Criterion benchmark for [`Detector::process_*`] on the threshold detector.
//!
//! Measures the full per-frame cost: mean intensity + state machine
//! transition + min-duration gate. Both `process_luma` and `process_rgb`
//! are covered so we can see the per-channel scan cost difference.
//!
//! Run with `cargo bench --bench threshold`.

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use scenesdetect::{
  frame::{LumaFrame, RgbFrame, Timebase, Timestamp},
  threshold::{Detector, Options},
};

fn make_buf(n: usize) -> Vec<u8> {
  let mut state: u32 = 0x9E3779B9;
  let mut buf = Vec::with_capacity(n);
  for _ in 0..n {
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    buf.push((state >> 24) as u8);
  }
  buf
}

fn bench_process_luma(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("threshold::Detector::process_luma");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let mut det = Detector::new(Options::default());
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = LumaFrame::new(&buf, w, h, w, Timestamp::new(pts, tb));
        pts += 33;
        black_box(det.process_luma(frame));
      });
    });
  }
  group.finish();
}

fn bench_process_rgb(c: &mut Criterion) {
  let tb = Timebase::new(1, NonZeroU32::new(1000).unwrap());
  let mut group = c.benchmark_group("threshold::Detector::process_rgb");
  for &(label, w, h) in &[
    ("720p", 1280u32, 720u32),
    ("1080p", 1920u32, 1080u32),
    ("4K", 3840u32, 2160u32),
  ] {
    let buf = make_buf((w * h * 3) as usize);
    group.throughput(criterion::Throughput::Bytes(buf.len() as u64));
    group.bench_function(label, |b| {
      let mut det = Detector::new(Options::default());
      let mut pts: i64 = 0;
      b.iter(|| {
        let frame = RgbFrame::new(&buf, w, h, w * 3, Timestamp::new(pts, tb));
        pts += 33;
        black_box(det.process_rgb(frame));
      });
    });
  }
  group.finish();
}

criterion_group!(benches, bench_process_luma, bench_process_rgb);
criterion_main!(benches);
