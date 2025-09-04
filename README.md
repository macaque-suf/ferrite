# WASM Audio Ferrite

High-performance, real-time audio noise reduction library powered by WebAssembly and Rust.

## Overview

WASM Audio Ferrite is a cutting-edge noise reduction system that brings professional-grade audio processing to the web. Built with Rust and compiled to WebAssembly, it delivers near-native performance with sub-millisecond latency.

## Key Features

- **🚀 Blazing Fast**: Sub-millisecond processing latency (0.055ms for 256 samples @ 48kHz)
- **🔧 Zero-Copy API**: Minimal memory overhead with direct WASM memory access
- **🎛️ Multi-Stage Processing**: Combines spectral subtraction and noise gating
- **🎵 Musical Noise Reduction**: Advanced Wiener filtering with comfort noise
- **📊 Adaptive Noise Profiling**: Learn and remove background noise in real-time
- **🌐 Cross-Platform**: Works in any modern web browser with WebAssembly support

## Performance Benchmarks

| Buffer Size | Sample Rate | Processing Time | CPU Usage | Real-time Factor |
|-------------|-------------|-----------------|-----------|------------------|
| 256 samples | 48 kHz      | 0.055ms        | <2%       | 96x              |
| 512 samples | 48 kHz      | 0.065ms        | <1.5%     | 164x             |
| 2048 samples| 48 kHz      | 0.175ms        | <0.5%     | 244x             |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wasm-audio-ferrite.git
cd wasm-audio-ferrite

# Install dependencies
npm install

# Build WASM module and TypeScript bindings
npm run build

# Run demos
npm run dev
```

### Basic Usage

```typescript
import { NoiseProcessor } from '@wasm-audio-ferrite/toolkit';

// Initialize the processor
const processor = await NoiseProcessor.create({
  sampleRate: 48000,
  useZeroCopy: true // Enable zero-copy API for best performance
});

// Configure noise reduction
processor.setReductionAmount(0.5); // 0.0 = minimal, 1.0 = maximum
processor.setGateThreshold(-40);   // dB
processor.setWienerMode(true);     // Enable Wiener filtering

// Process audio
const processedAudio = processor.processFrame(inputBuffer);
```

## Demos

The project includes several interactive demos:

1. **[Main Demo](demo/index.html)** - Complete noise reduction interface with file upload and real-time processing
2. **[Live Passthrough](demo/live-passthrough.html)** - Real-time microphone processing with performance metrics
3. **[Zero-Copy Benchmark](demo/test-zero-copy.html)** - Compare regular vs zero-copy API performance
4. **[Testing Suite](demo/test-noise-reduction.html)** - Advanced parameter tuning and stage-by-stage visualization

## Architecture

```
┌─────────────────────────────────────────────┐
│           Web Audio API (JavaScript)         │
├─────────────────────────────────────────────┤
│         TypeScript Wrapper (@toolkit)        │
├─────────────────────────────────────────────┤
│      WebAssembly Module (wasm-bindgen)      │
├─────────────────────────────────────────────┤
│         Rust Core (audio-processor)         │
│  ┌─────────────┐        ┌──────────────┐   │
│  │  Spectral   │        │   Noise      │   │
│  │ Subtraction │   ───> │    Gate      │   │
│  └─────────────┘        └──────────────┘   │
└─────────────────────────────────────────────┘
```

## Components

- **audio-processor-core/** - Rust implementation of noise reduction algorithms
- **packages/toolkit/** - TypeScript/JavaScript wrapper with Web Audio integration
- **demo/** - Interactive demonstrations and performance tests

## Building from Source

### Prerequisites

- Rust 1.70+ with `wasm32-unknown-unknown` target
- Node.js 18+
- wasm-pack (`cargo install wasm-pack`)

### Build Commands

```bash
# Build WASM module
npm run build:wasm

# Build TypeScript toolkit
npm run build:toolkit

# Build everything
npm run build

# Run development server
npm run dev
```

## API Documentation

### NoiseProcessor Class

The main interface for audio processing:

```typescript
class NoiseProcessor {
  // Create a new processor instance
  static async create(config: ProcessorConfig): Promise<NoiseProcessor>
  
  // Process audio frame (regular API)
  processFrame(input: Float32Array): Float32Array
  
  // Process audio frame (zero-copy API)
  processFrameZeroCopy(input: Float32Array): Float32Array
  
  // Learn noise profile from background audio
  learnNoise(samples: Float32Array): void
  
  // Configuration methods
  setReductionAmount(amount: number): void  // 0.0 - 1.0
  setGateThreshold(dbfs: number): void      // Typically -60 to 0
  setWienerMode(enabled: boolean): void
  setBypass(enabled: boolean): void
}
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) - WebAssembly bindings
- [Web Audio API](https://www.w3.org/TR/webaudio/) - Browser audio processing
- [rustfft](https://github.com/ejmahler/RustFFT) - Fast Fourier Transform implementation

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**WASM Audio Ferrite** - Professional audio processing for the modern web 🎵