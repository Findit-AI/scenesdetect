<div align="center">
<h1>scenesdetect</h1>
</div>
<div align="center">

A Rust port of [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — scene/shot cut detection built around a Sans-I/O streaming API, designed to slot in any other frame source.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/scenesdetect-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fscenesdetect" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/scenesdetect/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/scenesdetect?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-scenesdetect-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/scenesdetect?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/scenesdetect?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCwxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTZjLTEyLjExOC0xNy45MzctMzIuMjYyLTI4LjY0NS01My44ODItMjguNjQ1IGMtMzUuODMzLDAtNjQuOTg1LDI5LjE1Mi02NC45ODUsNjQuOTg2czI5LjE1Miw2NC45ODYsNjQuOTg1LDY0Ljk4NmMyMi4yODEsMCw0Mi43NTktMTEuMjE4LDU0Ljc3OC0zMC4wMDkgQzIwMC4yMDgsMTY5LjE0NywxOTguOTg1LDE2My41ODIsMTk0Ljc5OCwxNjAuOTAzeiIvPiA8L2c+IDwvZz4KDTwvc3ZnPg==" height="22">

</div>

## Overview

`scenesdetect` is a from-scratch Rust port of [PySceneDetect](https://github.com/Breakthrough/PySceneDetect). It is deliberately **Sans-I/O**: the crate never opens a file, decodes a packet, or spawns a thread. Callers hand frames in one by one, and each detector returns an `Option<Timestamp>` identifying the cut point — or nothing. Composing those point cuts into scene ranges is the caller's responsibility, which keeps this crate independent of any particular decoding pipeline.

Timestamps are represented as raw integer `pts + Timebase` (matching FFmpeg's `AVRational`) rather than floating-point seconds, so all arithmetic is exact and cross-stream comparisons are unambiguous.

## Detectors

| Module | Algorithm | Good for |
|---|---|---|
| [`histogram`] | YUV-luma histogram correlation | Generic cuts, robust to camera shake |
| [`phash`] | DCT-based perceptual hash (pHash) | Similarity-tolerant dedup / cut detection |
| [`threshold`] | Mean-brightness state machine | Fade-to-black / fade-in transitions |
| [`content`] | HSV-space delta + optional Canny edge delta | Motion/composition changes — the default PySceneDetect algorithm |
| [`adaptive`] | Rolling-average wrapper over `content` | Suppresses false positives on sustained fast motion |

[`histogram`]: https://docs.rs/scenesdetect/latest/scenesdetect/histogram/
[`phash`]: https://docs.rs/scenesdetect/latest/scenesdetect/phash/
[`threshold`]: https://docs.rs/scenesdetect/latest/scenesdetect/threshold/
[`content`]: https://docs.rs/scenesdetect/latest/scenesdetect/content/
[`adaptive`]: https://docs.rs/scenesdetect/latest/scenesdetect/adaptive/

## Features

- **Sans-I/O streaming API** — hand in `LumaFrame` / `RgbFrame` / `HsvFrame` (zero-copy slices), get `Option<Timestamp>` back per frame. No allocation on the hot path once the detector is primed.
- **Hand-written SIMD backends** — aarch64 NEON, x86 SSSE3 + AVX2 (runtime-dispatched via `is_x86_feature_detected!`), and wasm `simd128`. All with scalar fallbacks, toggleable per-detector via `Options::with_simd(false)`.
- **Exact rational timestamps** — `Timebase` mirrors FFmpeg's `AVRational`; `Timestamp` compares semantically across timebases via i128 cross-multiply.
- **`no_std` + `alloc`** — the crate builds without `std`; enable the default `std` feature for runtime x86 feature detection.
- **Optional `serde`** — all `Options` types derive `Serialize` / `Deserialize` under the `serde` feature.

## Installation

```toml
[dependencies]
scenesdetect = "0.1"
```

## Crate features

| Feature | Default | Purpose |
|---|---|---|
| `std` | ✓ | Runtime x86 SIMD dispatch, standard library types |
| `alloc` |   | `no_std` build using `alloc` only |
| `serde` |   | `Serialize` / `Deserialize` for all `Options` types |

## Benchmarks

Numbers below are per-frame runtimes from the [`benchmark.yml`](.github/workflows/benchmark.yml) CI workflow on GitHub-hosted runners, compiled with the default release profile (`opt-level = 3`, thin LTO). Each row is a single `process_*` call — that is, the full pipeline for one frame including the per-channel delta reduction. Lower is better; `fps` is `1 s / per-frame time`. Full data lives in the **Benchmarks** workflow artifacts.

### Per-detector timings at 1080p

Best SIMD-on path, single-threaded:

| Detector                               | macOS aarch64 NEON | Linux x86_64 AVX2 | Windows x86_64 AVX2 |
|---                                     |---:|---:|---:|
| `histogram`                            | 0.93 ms (≈1 080 fps) | 1.24 ms (≈810 fps)  | 1.26 ms (≈790 fps)  |
| `phash`                                | 1.65 ms (≈610 fps)   | 2.03 ms (≈490 fps)  | 2.22 ms (≈450 fps)  |
| `threshold` — luma                     | 0.12 ms (≈8 000 fps) | 0.33 ms (≈3 080 fps)| 0.34 ms (≈2 940 fps)|
| `threshold` — RGB                      | 0.38 ms (≈2 650 fps) | 0.98 ms (≈1 030 fps)| 0.99 ms (≈1 020 fps)|
| `content` — luma-only                  | 0.48 ms (≈2 080 fps) | 0.34 ms (≈2 940 fps)| 0.40 ms (≈2 510 fps)|
| `content` — BGR, no edges              | 3.38 ms (≈ 300 fps)  | 2.78 ms (≈360 fps)  | 2.84 ms (≈350 fps)  |
| `content` — BGR **with** Canny edges   | 58.0 ms (≈17 fps)    | 71.0 ms (≈14 fps)   | 75.8 ms (≈13 fps)   |
| `adaptive` — luma-only                 | 0.49 ms (≈2 040 fps) | 0.30 ms (≈3 300 fps)| 0.40 ms (≈2 500 fps)|
| `adaptive` — BGR, no edges             | 3.18 ms (≈ 315 fps)  | 2.78 ms (≈360 fps)  | 3.06 ms (≈325 fps)  |

### SIMD vs scalar at 1080p (`content::process_bgr`, default weights, no edges)

The BGR path is the hot spot — packed-BGR → planar HSV conversion is where the hand-written SIMD backends earn their keep. Scalar numbers come from the same benches with `Options::with_simd(false)`.

| Tier                                               | SIMD     | Scalar    | Uplift |
|---                                                 |---:|---:|---:|
| `macos-aarch64-neon`                               | 3.38 ms  | 4.61 ms   | **1.36×** |
| `ubuntu-x86_64-default` (runtime AVX2)             | 2.78 ms  | 24.99 ms  | **9.0×**  |
| `ubuntu-x86_64-native` (`-C target-cpu=native`)    | 2.72 ms  | 9.00 ms   | **3.3×**  |
| `ubuntu-x86_64-ssse3-only` (AVX/AVX2/FMA disabled) | 2.09 ms  | 21.34 ms  | **10.2×** |
| `windows-x86_64-default`                           | 2.84 ms  | 57.55 ms  | **20.3×** |

A few things fall out of this:

- **x86 SIMD is very much worth it.** Intel/AMD runners without the hand-written `std::arch` dispatch — i.e. scalar — run the BGR pipeline 9–20× slower than the SSSE3/AVX2 backend. The biggest x86 win is the 3-plane deinterleave via `PSHUFB`, which the compiler doesn't emit on its own.
- **NEON uplift is modest** because aarch64's auto-vectorizer handles the scalar fallback well; the hand-written NEON path still wins on the deinterleave (`vld3q_u8`) but the scalar baseline is already strong.
- **`-C target-cpu=native` closes most of the scalar gap** on x86 (9 ms vs 25 ms default scalar) by unlocking AVX2 for LLVM's auto-vectorizer, but it still loses to the hand-written dispatch by ~3×.
- **Canny edges are expensive.** Turning on `delta_edges` dominates the frame time at ~60–75 ms/1080p. Only enable it when color deltas aren't enough.
- **Adaptive overhead is ≈O(1) per frame.** Varying `window_width` from 1 to 16 moves the 1080p luma-only timing by <5% — the [rolling-sum fix](src/adaptive.rs) made the per-frame cost flat.

### Reproducing locally

```sh
cargo bench --bench content
cargo bench --bench adaptive
# ...or all of them:
cargo bench
```

The `benchmark.yml` workflow runs five matrix rows on every push to `main` and every PR touching `src/**`, `benches/**`, or the workflow file: `macos-aarch64-neon`, `ubuntu-x86_64-default`, `ubuntu-x86_64-native`, `ubuntu-x86_64-ssse3-only`, `windows-x86_64-default`. The per-run artifact contains both a bencher-format summary and the Criterion HTML detail tree.

## Acknowledgements

`scenesdetect` is a Rust port of [**PySceneDetect**](https://github.com/Breakthrough/PySceneDetect) by [Brandon Castellano](https://github.com/Breakthrough), released under the BSD 3-Clause license. The detector algorithms — histogram correlation, DCT-based pHash, brightness-threshold fades, HSV + Canny content deltas, and the rolling-average adaptive layer — are re-implementations of the algorithms described in PySceneDetect's source and documentation. Default parameters mirror PySceneDetect's where practical; any deliberate deviations are called out in the relevant module docs.

See [THIRD-PARTY.md](THIRD-PARTY.md) for the full upstream license text and additional third-party notices.

#### License

`scenesdetect` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT studio authors.

[Github-url]: https://github.com/findit-ai/scenesdetect/
[CI-url]: https://github.com/findit-ai/scenesdetect/actions/workflows/ci.yml
[doc-url]: https://docs.rs/scenesdetect
[crates-url]: https://crates.io/crates/scenesdetect
[codecov-url]: https://app.codecov.io/gh/findit-ai/scenesdetect/
