# WASM Audio Ferrite Demos

This directory contains demonstration applications for the WASM Audio Ferrite noise reduction system.

## Main Demos

### 1. [index.html](index.html) - **Main Demo Application**
- **Purpose**: Complete noise reduction demonstration
- **Features**: 
  - File upload and playback
  - Real-time noise reduction
  - Visual waveform and spectrum analysis
  - Parameter controls (gate threshold, reduction amount, etc.)
- **Best for**: General users wanting to try the noise reduction

### 2. [live-passthrough.html](live-passthrough.html) - **Live Audio Processing**
- **Purpose**: Real-time microphone processing demonstration
- **Features**:
  - Live microphone input with processed output
  - Real-time performance metrics (CPU, latency, dropouts)
  - Compare passthrough vs regular API vs zero-copy API
  - Visual level meters and waveform
- **Best for**: Testing real-time performance and latency

### 3. [test-zero-copy.html](test-zero-copy.html) - **Performance Benchmark**
- **Purpose**: Benchmark zero-copy API vs regular API
- **Features**:
  - Synthetic performance testing
  - Memory usage comparison
  - Processing time statistics (avg, min, max, P95)
  - Visual performance charts
- **Best for**: Developers evaluating performance improvements

### 4. [test-noise-reduction.html](test-noise-reduction.html) - **Noise Reduction Testing**
- **Purpose**: Comprehensive noise reduction parameter testing
- **Features**:
  - File upload with before/after comparison
  - Stage-by-stage processing visualization
  - Noise profile learning
  - Export processed audio
- **Best for**: Audio engineers tuning parameters

## Required Files

- `wasm/` - Contains the compiled WebAssembly module
  - `audio_processor_core_bg.wasm` - The WASM binary
  - `audio_processor_core.js` - JavaScript bindings
  - `audio_processor_core.d.ts` - TypeScript definitions

## Usage

1. **Local Development**: 
   ```bash
   # Serve the demo directory with a local server
   python3 -m http.server 8000
   # or
   npx serve .
   ```

2. **Testing**: Open `http://localhost:8000/` in your browser

## Browser Requirements

- Modern browser with WebAssembly support
- Web Audio API support
- For live audio: Microphone permissions
- For best performance metrics: Chrome/Edge (for performance.memory API)

## Performance Expectations

With zero-copy API at 48kHz sample rate:
- **256 samples buffer**: ~0.05-0.1ms processing time (<2% CPU)
- **512 samples buffer**: ~0.05-0.15ms processing time (<1.5% CPU)  
- **2048 samples buffer**: ~0.1-0.2ms processing time (<0.5% CPU)

## Development Notes

For development and debugging, additional test files are available in the repository history but have been removed from the main branch to keep the demo directory clean and focused.