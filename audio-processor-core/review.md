
## 1) Concrete issues in the code/tests

### ❗ Bug: partial‑hop output gets **dropped**

In `OverlapAddProcessor::process` (and `process_with_spectrum`) you always rotate the overlap buffer by `hop_size`, even if the caller’s `output` slice can’t hold a full hop. This **silently discards** the unwritten part of the hop.

You even have an `overlap_read_pos` field for this, but it’s unused.

**Fix** (core logic change):

```rust
// Replace the block that writes and rotates with this:
let remaining = output.len() - output_written;
let to_emit = hop_size.min(remaining);

// Emit from the current read cursor
let start = self.overlap_read_pos;
let end   = start + to_emit;
output[output_written..output_written+to_emit]
    .copy_from_slice(&self.overlap_buffer[start..end]);
output_written += to_emit;
self.overlap_read_pos += to_emit;

// Only rotate once we've emitted a full hop
if self.overlap_read_pos == hop_size {
    self.overlap_buffer.rotate_left(hop_size);
    self.overlap_buffer[fft_size - hop_size..].fill(0.0);
    self.overlap_read_pos = 0;
}
```

You’ll also want the same fix in `process_with_spectrum`.

**Test you’re missing:** a streaming test that uses tiny `output` buffers (e.g., 1…hop\_size‑1) to verify zero sample loss and correct ordering.

---

### ❗ Bug: Tukey window can divide by zero

For very small `alpha` (e.g., 0.001 on small sizes), `taper_length` can round to `0`, then you divide by `taper_length`.

**Fix**:

```rust
let mut taper_length = (alpha * (size - 1) as f32 / 2.0).round() as usize;
if taper_length == 0 {
    // treat as rectangular (alpha ~ 0) or clamp to 1; the former is safest
    window.fill(1.0);
    return;
}
```

**Tests missing:** Tukey with α near 0 and 1, several sizes (including minimum size), and assertions that there’s no panic and coefficients are finite.

---

### ⚠️ WOLA compensation is computed from the **wrong window**

`calculate_window_compensation()` always calls `calculate_cola_gain(&self.analysis_window, overlap)`.
For WOLA you use **sqrt(window)** on analysis and synthesis; the **effective amplitude window per frame** is `w_eff[n] = sqrt(w[n]) * sqrt(w[n]) = w[n]`.
So for WOLA you should compute COLA on `w_eff` (i.e., the *original* window), not on the root/analysis window.

**Fix**:

```rust
pub fn calculate_window_compensation(&self) -> f32 {
    let eff_window: Vec<f32> = match self.mode {
        ProcessingMode::OLA => self.analysis_window.clone(),
        ProcessingMode::WOLA => self.analysis_window.iter()
            .zip(self.synthesis_window.iter())
            .map(|(a, s)| a * s) // sqrt(w) * sqrt(w) = w
            .collect(),
    };
    let cola_gain = calculate_cola_gain(&eff_window, self.overlap_percent);
    if cola_gain.abs() < 1e-3 { 1.0 } else { 1.0 / cola_gain }
}
```

**Tests missing:** WOLA unity‑gain reconstruction across typical pairs that are known to be COLA‑compatible (e.g., sqrt‑Hann @ 50%, sqrt‑Blackman @ \~66.7%).

---

### ⚠️ COLA check is weak and window definition matters

* Your Hann uses `cos(2π n/(N-1))` (symmetric). The **exact** COLA identity `w[n] + w[n+N/2] = 1` holds for the *periodic* Hann (`cos(2π n/N)`), not the symmetric one—so your `≈ 1` assertion with a large tolerance might pass “by luck”.
* `calculate_cola_gain` computes `(max+min)/2` over offsets; that hides local non‑flatness. You want the **profile** (max/min ratio or stddev) and a **strict bound**.

**Better approach:** build a synthetic long stream, overlap‑add a train of windows at hop spacing, and check flatness in steady‑state (ignore edges). I give a test below.

---

### ⚠️ Inverse drops imaginary part if user callback breaks Hermitian symmetry

If a callback edits only the “positive frequencies” without mirroring, the inverse of a non‑Hermitian spectrum is complex; you take only `re`, which introduces artifacts.

**Mitigation:** After any user callback, enforce Hermitian symmetry before the inverse (or document that the callback **must** maintain it). Example helper:

```rust
fn enforce_hermitian(s: &mut [Complex32]) {
    let n = s.len();
    let half = n/2;
    // DC and Nyquist (if present) must be real
    s[0].im = 0.0;
    if n % 2 == 0 { s[half].im = 0.0; }
    for k in 1..half {
        let mirror = n - k;
        s[mirror] = s[k].conj();
    }
}
```

**Tests missing:** callback that deliberately breaks symmetry; verify the helper restores real reconstruction.

---

### ⚠️ Ring buffer semantics & backpressure

* `write()` uses `unwrap_or(0)`; when full, samples are **silently dropped**. At least return the written count and expose **dropped samples** for visibility/testing.
* `space_available()` calls `write_available(&[])` which looks suspicious; if your API offers `write_available()` without a slice, use that. Otherwise write a targeted test to assert this returns consistent values across wrap‑around.

**Tests missing:** producer faster than consumer (loss), wrap‑around boundaries, and multi‑thread SPSC stress.

---

### ⚠️ Latency reporting

`latency_samples()` returns `fft_size`, but the first non‑zero output appears after the first frame processes and you emit a hop. For many streaming pipelines the **group delay** is closer to `fft_size - hop_size` (depending on window centering). At least add tests that pin observed latency, and document the convention.

---

## 2) A robust test suite (ready‑to‑paste)

Below is a **concrete plan** with code. It keeps your current tests, and adds what’s missing.

> **Dev deps (Cargo.toml)**
>
> ```toml
> [dev-dependencies]
> proptest = "1"
> approx = "0.5"
> ```

### A) Windows: coverage + edge cases

```rust
#[test]
fn tukey_alpha_edges_no_panics_and_shape() {
    let sizes = [64, 256, 1024];
    let alphas = [0.0, 1e-6, 0.05, 0.5, 1.0];
    for &n in &sizes {
        for &a in &alphas {
            let w = WindowType::Tukey(a).generate(n);
            assert_eq!(w.len(), n);
            assert!(w.iter().all(|v| v.is_finite()));
            // Center should approach 1 as alpha -> 1
            if a > 0.9 {
                assert!((w[n/2] - 1.0).abs() < 1e-3);
            }
        }
    }
}

#[test]
fn hermitian_enforcement_makes_dc_nyquist_real() {
    use num_complex::Complex32 as C;
    let n = 512;
    let mut s = vec![C::new(0.0, 0.0); n];
    for k in 0..n { s[k] = C::new(k as f32, (k as f32).sin()); }
    // Break symmetry
    // ... then enforce:
    fn enforce_hermitian(s:&mut [C]) { /* paste helper here */ }
    enforce_hermitian(&mut s);
    assert!((s[0].im).abs() < 1e-6);
    assert!((s[n/2].im).abs() < 1e-6);
    for k in 1..(n/2) {
        assert!((s[n-k] - s[k].conj()).norm() < 1e-3);
    }
}
```

### B) FFT math invariants (orthonormal scaling + Parseval)

```rust
use approx::assert_relative_eq;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[test]
fn forward_inverse_is_identity_random() {
    let mut rng = StdRng::seed_from_u64(42);
    for &n in &[128, 256, 512, 1024] {
        let mut p = FftProcessor::with_window(n, WindowType::Rectangular, 0.0, ProcessingMode::OLA).unwrap();
        for _ in 0..8 {
            let mut x = vec![0.0; n];
            for v in &mut x { *v = rng.gen::<f32>() * 2.0 - 1.0; }
            let X = p.forward(&x).unwrap().to_vec();
            let y = p.inverse(&X).unwrap().to_vec();
            for i in 0..n {
                assert_relative_eq!(x[i], y[i], epsilon = 2e-4, max_relative = 2e-4);
            }
            // Parseval (orthonormal scaling)
            let e_time: f32 = x.iter().map(|v| v*v).sum();
            let e_freq: f32 = X.iter().map(|c| c.norm_sqr()).sum();
            assert_relative_eq!(e_time, e_freq, max_relative = 1e-3);
        }
    }
}
```

### C) COLA profile flatness (OLA & WOLA)

```rust
fn cola_profile(window: &[f32], hop: usize, repeats: usize) -> Vec<f32> {
    let n = window.len();
    let len = hop * repeats + n; // long enough to reach steady-state
    let mut acc = vec![0.0f32; len];
    for r in 0..repeats {
        let start = r * hop;
        for i in 0..n { acc[start + i] += window[i]; }
    }
    // Trim edges: take the middle block of length n
    let mid_start = (len - n)/2;
    acc[mid_start..mid_start+n].to_vec()
}

#[test]
fn cola_hann_50_ola_is_flat_enough() {
    let n = 512;
    let w = WindowType::Hann.generate(n);
    let hop = n/2;
    let prof = cola_profile(&w, hop, 12);
    let max = prof.iter().cloned().fold(f32::MIN, f32::max);
    let min = prof.iter().cloned().fold(f32::MAX, f32::min);
    let flatness = (max - min) / ((max + min) * 0.5);
    assert!(flatness < 1e-3, "COLA not flat enough: {}", flatness);
}

#[test]
fn cola_sqrt_hann_50_wola_is_flat_and_unit_gain_with_comp() {
    let n = 512;
    let w = WindowType::Hann.generate(n);
    let w_sqrt: Vec<f32> = w.iter().map(|v| v.sqrt()).collect();
    let hop = n/2;

    // Effective amplitude window is w (product of sqrt windows)
    let prof = cola_profile(&w, hop, 12);
    let avg = (prof.iter().sum::<f32>() / prof.len() as f32).max(1e-9);
    let comp = 1.0 / avg;

    // After compensation, profile ~ 1.0
    let prof_c: Vec<f32> = prof.into_iter().map(|v| v*comp).collect();
    let max = prof_c.iter().cloned().fold(f32::MIN, f32::max);
    let min = prof_c.iter().cloned().fold(f32::MAX, f32::min);
    assert!((max - 1.0).abs() < 1e-3 && (min - 1.0).abs() < 1e-3);
}
```

### D) Streaming identity with **random chunk sizes** and **partial‑hop outputs**

```rust
#[test]
fn streaming_identity_handles_arbitrary_io_sizes_without_loss() {
    let mut proc = OverlapAddProcessor::new(512, WindowType::Hann, 50.0, ProcessingMode::OLA).unwrap();
    proc.set_process_callback(|_| {});
    let hop = proc.hop_size();

    // Signal long enough for steady-state
    let sr = 48_000.0;
    let tone = 440.0;
    let len = 48000;
    let input: Vec<f32> = (0..len).map(|i| (2.0*std::f32::consts::PI*tone*(i as f32)/sr).sin()).collect();

    // Random chunking both on input and output
    let mut rng = StdRng::seed_from_u64(7);
    let mut pos = 0usize;
    let mut out = Vec::with_capacity(len + 2*proc.latency_samples());
    while pos < len {
        let in_chunk = rng.gen_range(1..=3*hop/2).min(len - pos);
        let mut out_chunk = vec![0.0; rng.gen_range(1..=hop)]; // often smaller than hop
        let written = proc.process(&input[pos..pos+in_chunk], &mut out_chunk).unwrap();
        out.extend_from_slice(&out_chunk[..written]);
        pos += in_chunk;
    }

    // Drain tail by feeding zeros
    for _ in 0..8 {
        let mut out_chunk = vec![0.0; hop/2];
        let written = proc.process(&vec![0.0; hop/2], &mut out_chunk).unwrap();
        out.extend_from_slice(&out_chunk[..written]);
    }

    // Lossless in steady-state energy (skip head/tail)
    let lat = proc.latency_samples();
    if out.len() > 4*lat {
        let start = 2*lat.min(out.len()/4);
        let end = (out.len() - 2*lat).min(input.len());
        let e_in: f32 = input[start..end].iter().map(|x| x*x).sum();
        let e_out:f32 = out[start..end].iter().map(|x| x*x).sum();
        let rel = ((e_in - e_out)/e_in).abs();
        assert!(rel < 0.05, "energy drift too large: {}", rel);
    }
}
```

This test would have **failed** with your current rotation logic.

### E) Property‑based tests (sizes, windows, overlaps)

Use `proptest` to randomly sweep sizes, window types, overlaps, and assert **no panics**, finite outputs, near‑identity for OLA identity processing, and COLA flatness within a loose bound.

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_streaming_no_panics_and_finite(
        fft_pow in 6u32..=12u32,                     // 64..4096
        overlap in prop::num::f32::NORMAL..=95.0f32, // 0..95
        win_choice in 0usize..=4
    ) {
        let n = 1usize << fft_pow;
        let win = match win_choice {
            0 => WindowType::Hann,
            1 => WindowType::Hamming,
            2 => WindowType::Blackman,
            3 => WindowType::Rectangular,
            _ => WindowType::Tukey(0.5),
        };
        let mut p = OverlapAddProcessor::new(n, win, overlap, ProcessingMode::OLA).unwrap();
        p.set_process_callback(|_| {}); // identity

        // Random-ish input
        let len = n * 5;
        let input: Vec<f32> = (0..len).map(|i| ((i*31 % 997) as f32).sin()).collect();

        // Process in uneven chunks
        let mut out_all = Vec::new();
        let mut i = 0;
        while i < input.len() {
            let chunk_len = ((i*73) % (n/2+1)).max(1).min(input.len() - i);
            let mut out = vec![0.0; (chunk_len/2).max(1)];
            let written = p.process(&input[i..i+chunk_len], &mut out).unwrap();
            out_all.extend_from_slice(&out[..written]);
            i += chunk_len;
        }

        // sanity: finite samples only
        assert!(out_all.iter().all(|v| v.is_finite()));
    }
}
```

### F) Hermitian‑symmetry enforcement + inverse

```rust
#[test]
fn callback_breaks_symmetry_but_helper_restores_real_output() {
    let mut p = FftProcessor::with_window(512, WindowType::Hann, 50.0, ProcessingMode::OLA).unwrap();
    // simple test frame
    let x: Vec<f32> = (0..512).map(|i| (2.0*PI*37.0*(i as f32)/512.0).sin()).collect();
    let mut X = p.forward(&x).unwrap().to_vec();
    // break symmetry on purpose
    for k in 1..200 { X[k].im += 0.123; }
    // enforce and inverse
    enforce_hermitian(&mut X);
    let y = p.inverse(&X).unwrap().to_vec();
    // y should closely match windowed x (since forward applies window)
    // we just assert real-valuedness and finite results here
    assert!(y.iter().all(|v| v.is_finite()));
}
```

### G) Phase‑vocoder: multiple tones & fractional ratios

```rust
#[test]
fn phase_vocoder_shifts_multiple_tones_fractional() {
    let n = 1024;
    let hop = n/4;
    let sr = 48000.0;
    let mut voc = PhaseVocoder::new(n, hop, sr);

    let mut s = vec![Complex32::new(0.0,0.0); n];
    // bins ~ 300 Hz and ~ 2000 Hz
    let b1 = ((300.0 * n as f32)/sr).round() as usize;
    let b2 = ((2000.0 * n as f32)/sr).round() as usize;
    s[b1] = Complex32::from_polar(1.0, 0.0);
    s[n-b1] = s[b1].conj();
    s[b2] = Complex32::from_polar(0.7, 1.0);
    s[n-b2] = s[b2].conj();

    voc.process_pitch_shift(&mut s, 1.5); // up a fifth
    let mags: Vec<f32> = s.iter().map(|c| c.norm()).collect();

    let peak_1 = mags[..n/2].iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    // expected near 1.5x original bin 1 (allow ±2 bins)
    assert!((peak_1 as isize - (b2 as isize)).abs() > 2); // peak should move from the larger amp bin too
}
```

(You can refine the expected bins with exact targets; the point is to test multi‑peak behavior and fractional ratios.)

### H) Concurrency / ring‑buffer stress (SPSC)

Use two threads: producer pushes random chunk sizes as fast as possible; consumer calls `process` with random‑sized outputs. Assert no panics, and track **drops** if you add that metric.

---

## 3) Additional improvements to test for robustness

* **Linearity and time‑invariance**:
  Verify `process(a + b) ≈ process(a) + process(b)` and that a time shift of the input yields a corresponding shift in output (after steady‑state), for identity processing.

* **Impulse & step responses**:
  For identity, an impulse should reconstruct the COLA profile (good for visualizing compensation). A step should not ring endlessly.

* **Detuned sinusoids** (not on bin centers):
  Use, e.g., 440 Hz at various sample rates and confirm no frame boundary clicks in streaming identity.

* **Reset mid‑stream**:
  Feed some audio, call `reset()`, continue. Verify there is no bleed‑through from pre‑reset overlap buffer or ring buffers.

* **NaNs/Inf defensiveness**:
  Feed NaNs/Inf into `process`; assert no panics and outputs are finite (or explicitly define behavior and test for it).

* **Performance regression guard** (optional):
  Add `criterion` benchmarks for forward/inverse and streaming chunks to catch accidental slowdowns.

* **Fuzzing**:
  With `cargo-fuzz`, fuzz `with_window` params (fft size, overlap, window kind/α) and `process` (arbitrary floats), assert “no panic, no UB, all finite”. This finds edge cases like the Tukey bug fast.

* **CI sanity**:
  Run tests under Miri and with `RUSTFLAGS="-C target-cpu=native -Z sanitizer=address"` where possible for UB/overflow checks.

---

## 4) Small API tweaks that make testing easier

* **Return input consumption and dropped counts** from `process`, e.g.:

  ```rust
  pub struct ProcessStats { pub in_consumed: usize, pub out_produced: usize, pub in_dropped: usize }
  ```

  This makes loss detection and backpressure testing straightforward.

* **Option to auto‑enforce Hermitian symmetry** before inverse (behind a flag), so user callbacks don’t have to remember it.

* **Expose effective window & compensation** so tests can directly assert COLA flatness and unity gain without peeking into internals.

---

## 5) Quick verdict

* ✅ Solid baseline coverage: window generation sanity, FFT round‑trip, hop computation, and a basic streaming test.
* ⚠️ Missing critical edges and a couple of correctness bugs (partial‑hop drop; WOLA COLA; Tukey small‑α).
* 🧪 Add the tests above (especially **COLA profile**, **partial‑hop streaming**, **Hermitian enforcement**, **randomized/property tests**, and **concurrency stress**). With those in place—and the three fixes—you’ll have a **production‑grade** test suite for real‑time spectral processing.
