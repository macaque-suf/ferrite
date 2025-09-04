# Audio Processor Core

High-performance audio processing library for real-time noise reduction, written in Rust and compiled to WebAssembly.

## Features

### 🎯 Core Components

- **Noise Gate** - Advanced real-time gating with:
  - Smooth envelope following with separate attack/release times
  - Hold time to prevent chattering
  - Lookahead for anticipating transients
  - Soft knee for transparent gating
  - Hysteresis to prevent rapid state changes
  - Variable ratio for expansion vs hard gating

- **FFT Processor** - Spectral processing with:
  - Multiple window types (Hann, Hamming, Blackman, etc.)
  - OLA/WOLA processing modes
  - Optimized for real-time performance
  - Phase vocoder support

- **Noise Profile** - Intelligent noise analysis:
  - Adaptive noise estimation
  - Speech activity detection
  - Spectral smoothing
  - Multi-band processing

- **Ring Buffer** - Lock-free audio streaming:
  - Thread-safe SPSC (Single Producer Single Consumer)
  - Zero-copy operations where possible
  - Optimized for audio workloads

## 🚀 Getting Started

### Building for WebAssembly

```bash
# Install wasm-pack if you haven't already
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the WASM module
wasm-pack build --target web

# For optimized production build
wasm-pack build --target web --release
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_noise_gate_creation
```

### Benchmarks

```bash
# Run benchmarks
cargo bench
```

## 📊 Performance

The library is optimized for real-time audio processing with:
- SIMD optimizations where available
- Lock-free data structures for audio streaming
- Minimal allocations in hot paths
- Cache-friendly memory layouts

### Typical Performance Metrics

| Component | Latency | CPU Usage |
|-----------|---------|-----------|
| Noise Gate | < 1ms | ~2% @ 48kHz |
| FFT (512 samples) | < 2ms | ~5% @ 48kHz |
| Ring Buffer | < 0.1ms | < 1% |

## 🔧 Configuration

### Noise Gate Example

```rust
use audio_processor_core::noise_gate::{NoiseGate, NoiseGateConfig};

// Create configuration
let mut config = NoiseGateConfig::with_sample_rate(48000.0);
config.threshold_db = -40.0;
config.ratio = 10.0;
config.attack_ms = 5.0;
config.release_ms = 100.0;

// Create and use gate
let mut gate = NoiseGate::new(config);
gate.process_mono(&mut audio_buffer);
```

### FFT Processing Example

```rust
use audio_processor_core::fft_processor::{FftProcessor, WindowType, ProcessingMode};

// Create FFT processor
let processor = FftProcessor::with_window(
    512,                        // FFT size
    WindowType::Hann,          // Window type
    75.0,                      // Overlap percent
    ProcessingMode::WOLA       // Processing mode
).unwrap();
```

## 🏗️ Architecture

The library follows a modular architecture:

```
audio-processor-core/
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── noise_gate.rs       # Noise gate implementation
│   ├── fft_processor.rs    # FFT and spectral processing
│   ├── noise_profile.rs    # Noise profiling and analysis
│   ├── ring_buffer.rs      # Lock-free ring buffer
│   └── utils.rs            # Common utilities and DSP helpers
├── tests/
│   └── web.rs              # WebAssembly tests
├── Cargo.toml              # Dependencies and configuration
└── README.md               # This file
```

## 🔐 Safety

The library uses `unsafe` code in performance-critical sections, particularly in:
- Ring buffer operations for lock-free concurrency
- SIMD operations for vectorized processing
- Memory management for zero-copy operations

All unsafe code is carefully documented and tested.

## 📝 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE_APACHE))
- MIT license ([LICENSE-MIT](LICENSE_MIT))

at your option.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`cargo test`)
2. No new warnings (`cargo build --release`)
3. Code follows Rust idioms and best practices
4. Performance-critical changes include benchmarks

## 📚 Documentation

For detailed API documentation, run:

```bash
cargo doc --open
```

## 🎯 Roadmap

- [ ] Add more noise reduction algorithms (spectral subtraction, Wiener filter)
- [ ] Implement adaptive filtering
- [ ] Add support for multi-channel processing
- [ ] GPU acceleration via WebGPU
- [ ] Real-time visualization components