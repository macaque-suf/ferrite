/**
 * Type definitions for Web Audio Toolkit
 * 
 * This module provides comprehensive TypeScript types for the audio processing
 * toolkit, ensuring type safety across the JavaScript/WASM boundary.
 */

// ============================================================================
// Basic Audio Types
// ============================================================================

/**
 * Supported audio sample rates in Hz
 */
export type SampleRate = 44100 | 48000 | 88200 | 96000;

/**
 * Audio frame size options (must be power of 2)
 */
export type FrameSize = 128 | 256 | 512 | 1024 | 2048 | 4096;

/**
 * Window function types for spectral analysis
 */
export type WindowFunction = 'hann' | 'hamming' | 'blackman' | 'rectangular';

/**
 * Audio channel configuration
 */
export type ChannelConfig = 'mono' | 'stereo';

/**
 * Processing quality presets
 */
export type QualityPreset = 'low' | 'medium' | 'high' | 'ultra';

// ============================================================================
// Noise Gate Types
// ============================================================================

/**
 * Configuration for the noise gate processor
 */
export interface NoiseGateConfig {
  /** Threshold in decibels below which signal is gated */
  threshold_db: number;
  
  /** Attack time in milliseconds (gate opening speed) */
  attack_ms: number;
  
  /** Release time in milliseconds (gate closing speed) */
  release_ms: number;
  
  /** Hold time in milliseconds (minimum open time) */
  hold_ms?: number;
  
  /** Lookahead time in milliseconds for smoother transitions */
  lookahead_ms?: number;
  
  /** Reduction ratio (0.0 = complete gate, 1.0 = no gating) */
  ratio?: number;
  
  /** Enable soft knee for smoother transitions */
  soft_knee?: boolean;
  
  /** Soft knee width in dB */
  knee_width_db?: number;
}

/**
 * Noise gate state for monitoring
 */
export interface NoiseGateState {
  /** Current gate state */
  is_open: boolean;
  
  /** Current envelope level in dB */
  envelope_db: number;
  
  /** Current gain reduction in dB */
  gain_reduction_db: number;
  
  /** Time since last state change in ms */
  time_since_change_ms: number;
}

// ============================================================================
// Spectral Subtraction Types
// ============================================================================

/**
 * Configuration for spectral subtraction noise reduction
 */
export interface SpectralSubtractionConfig {
  /** FFT size for frequency analysis */
  fft_size: FrameSize;
  
  /** Overlap percentage (0-100) for overlap-add processing */
  overlap_percent?: number;
  
  /** Window function for FFT */
  window_function?: WindowFunction;
  
  /** Noise oversubtraction factor (typically 1.0-3.0) */
  alpha?: number;
  
  /** Spectral floor parameter to prevent over-subtraction (0.0-1.0) */
  beta?: number;
  
  /** Smoothing factor for noise profile updates (0.0-1.0) */
  noise_smoothing?: number;
  
  /** Enable adaptive noise profiling */
  adaptive?: boolean;
  
  /** Minimum noise floor in dB */
  noise_floor_db?: number;
}

/**
 * Noise profile for spectral subtraction
 */
export interface NoiseProfile {
  /** Frequency bin magnitudes representing noise spectrum */
  spectrum: Float32Array;
  
  /** Number of frames used to estimate this profile */
  frame_count: number;
  
  /** Timestamp when profile was last updated */
  timestamp: number;
  
  /** Is this profile locked (manual) or adaptive */
  is_locked: boolean;
  
  /** Statistical confidence (0.0-1.0) */
  confidence?: number;
}

/**
 * Spectral analysis data for visualization
 */
export interface SpectralData {
  /** Frequency bins in Hz */
  frequencies: Float32Array;
  
  /** Magnitude spectrum in dB */
  magnitudes: Float32Array;
  
  /** Phase spectrum in radians (optional) */
  phases?: Float32Array;
  
  /** Sample rate used for analysis */
  sample_rate: number;
  
  /** FFT size used */
  fft_size: number;
}

// ============================================================================
// Ring Buffer Types
// ============================================================================

/**
 * Ring buffer statistics for performance monitoring
 */
export interface RingBufferStats {
  /** Total samples written */
  total_written: number;
  
  /** Total samples read */
  total_read: number;
  
  /** Number of overflow events */
  overflow_count: number;
  
  /** Number of underflow events */
  underflow_count: number;
  
  /** Current buffer fill level (0.0-1.0) */
  fill_level: number;
  
  /** Average latency in samples */
  average_latency?: number;
}

/**
 * Ring buffer configuration
 */
export interface RingBufferConfig {
  /** Buffer size in samples (will be rounded to power of 2) */
  size: number;
  
  /** Enable statistics tracking */
  track_stats?: boolean;
  
  /** Initial fill level before processing starts (0.0-1.0) */
  prefill_level?: number;
}

// ============================================================================
// Processing Chain Types
// ============================================================================

/**
 * Overall processor configuration
 */
export interface ProcessorConfig {
  /** Sample rate in Hz */
  sample_rate: SampleRate;
  
  /** Processing frame size */
  frame_size: FrameSize;
  
  /** Enable noise gate */
  noise_gate?: NoiseGateConfig | boolean;
  
  /** Enable spectral subtraction */
  spectral_subtraction?: SpectralSubtractionConfig | boolean;
  
  /** Quality preset (overrides individual settings) */
  quality?: QualityPreset;
  
  /** Enable worker thread processing */
  use_worker?: boolean;
  
  /** Worker thread path (if custom) */
  worker_path?: string;
  
  /** Enable SIMD optimizations if available */
  use_simd?: boolean;
  
  /** Maximum processing latency in ms */
  max_latency_ms?: number;
}

/**
 * Processing statistics
 */
export interface ProcessingStats {
  /** Average processing time per frame in ms */
  avg_processing_time_ms: number;
  
  /** Peak processing time in ms */
  peak_processing_time_ms: number;
  
  /** Total frames processed */
  frames_processed: number;
  
  /** Total processing time in ms */
  total_processing_time_ms: number;
  
  /** Current CPU usage percentage (0-100) */
  cpu_usage_percent?: number;
  
  /** Ring buffer statistics */
  buffer_stats?: RingBufferStats;
  
  /** Noise gate state */
  gate_state?: NoiseGateState;
}

// ============================================================================
// Audio Buffer Types
// ============================================================================

/**
 * Enhanced audio buffer with metadata
 */
export interface EnhancedAudioBuffer {
  /** The actual AudioBuffer */
  buffer: AudioBuffer;
  
  /** Sample rate */
  sample_rate: number;
  
  /** Number of channels */
  channel_count: number;
  
  /** Duration in seconds */
  duration: number;
  
  /** Peak amplitude (0.0-1.0) */
  peak_amplitude?: number;
  
  /** RMS level in dB */
  rms_db?: number;
  
  /** Processing metadata */
  metadata?: ProcessingMetadata;
}

/**
 * Processing metadata attached to audio buffers
 */
export interface ProcessingMetadata {
  /** Noise reduction amount applied in dB */
  noise_reduction_db?: number;
  
  /** Gate reduction applied in dB */
  gate_reduction_db?: number;
  
  /** Processing timestamp */
  processed_at: number;
  
  /** Processing chain used */
  processing_chain: string[];
  
  /** Quality preset used */
  quality?: QualityPreset;
}

// ============================================================================
// WebAssembly Interface Types
// ============================================================================

/**
 * WASM module initialization options
 */
export interface WasmInitOptions {
  /** Path to WASM module */
  wasmPath?: string;
  
  /** Memory configuration */
  memory?: {
    initial: number;
    maximum?: number;
    shared?: boolean;
  };
  
  /** Enable debug mode */
  debug?: boolean;
  
  /** SIMD support detection */
  simd?: 'auto' | 'force' | 'disable';
}

/**
 * WASM processor interface
 */
export interface WasmProcessor {
  /** Initialize the processor */
  initialize(config: ProcessorConfig): Promise<void>;
  
  /** Process audio frame */
  process(input: Float32Array): Float32Array;
  
  /** Update configuration */
  updateConfig(config: Partial<ProcessorConfig>): void;
  
  /** Get current statistics */
  getStats(): ProcessingStats;
  
  /** Reset processor state */
  reset(): void;
  
  /** Cleanup resources */
  destroy(): void;
}

// ============================================================================
// Callback Types
// ============================================================================

/**
 * Audio processing callback
 */
export type ProcessCallback = (
  inputBuffer: Float32Array,
  outputBuffer: Float32Array,
  frameTime: number
) => void;

/**
 * Noise detection callback
 */
export type NoiseDetectionCallback = (
  noiseLevel: number,
  timestamp: number
) => void;

/**
 * Error callback
 */
export type ErrorCallback = (
  error: ProcessingError,
  canRecover: boolean
) => void;

/**
 * Statistics update callback
 */
export type StatsCallback = (stats: ProcessingStats) => void;

// ============================================================================
// Error Types
// ============================================================================

/**
 * Processing error types
 */
export enum ProcessingErrorType {
  BUFFER_OVERFLOW = 'BUFFER_OVERFLOW',
  BUFFER_UNDERFLOW = 'BUFFER_UNDERFLOW',
  INITIALIZATION_FAILED = 'INITIALIZATION_FAILED',
  WASM_ERROR = 'WASM_ERROR',
  INVALID_CONFIG = 'INVALID_CONFIG',
  AUDIO_CONTEXT_ERROR = 'AUDIO_CONTEXT_ERROR',
  WORKER_ERROR = 'WORKER_ERROR',
  MEMORY_ERROR = 'MEMORY_ERROR',
  UNKNOWN = 'UNKNOWN'
}

/**
 * Processing error with details
 */
export interface ProcessingError {
  /** Error type */
  type: ProcessingErrorType;
  
  /** Human-readable message */
  message: string;
  
  /** Error details */
  details?: Record<string, any>;
  
  /** Stack trace if available */
  stack?: string;
  
  /** Timestamp */
  timestamp: number;
  
  /** Can processing continue? */
  recoverable: boolean;
}

// ============================================================================
// Web Audio API Extensions
// ============================================================================

/**
 * Audio worklet processor options
 */
export interface AudioWorkletOptions {
  /** Processor name */
  processorName: string;
  
  /** Number of inputs */
  numberOfInputs?: number;
  
  /** Number of outputs */
  numberOfOutputs?: number;
  
  /** Output channel count */
  outputChannelCount?: number[];
  
  /** Parameter descriptors */
  parameterDescriptors?: AudioParamDescriptor[];
  
  /** Processor options */
  processorOptions?: ProcessorConfig;
}

/**
 * Audio parameter descriptor
 */
export interface AudioParamDescriptor {
  /** Parameter name */
  name: string;
  
  /** Default value */
  defaultValue: number;
  
  /** Minimum value */
  minValue?: number;
  
  /** Maximum value */
  maxValue?: number;
  
  /** Automation rate */
  automationRate?: 'a-rate' | 'k-rate';
}

// ============================================================================
// Visualization Types
// ============================================================================

/**
 * Visualization configuration
 */
export interface VisualizationConfig {
  /** Visualization type */
  type: 'waveform' | 'spectrum' | 'spectrogram' | 'level_meter';
  
  /** Update rate in Hz */
  update_rate?: number;
  
  /** FFT size for spectrum/spectrogram */
  fft_size?: FrameSize;
  
  /** Color scheme */
  color_scheme?: 'default' | 'viridis' | 'plasma' | 'inferno' | 'grayscale';
  
  /** Frequency scale */
  frequency_scale?: 'linear' | 'log' | 'mel';
  
  /** Amplitude scale */
  amplitude_scale?: 'linear' | 'log';
  
  /** Min/max frequency range */
  frequency_range?: [number, number];
  
  /** Min/max amplitude range in dB */
  amplitude_range?: [number, number];
}

/**
 * Visualization data frame
 */
export interface VisualizationFrame {
  /** Frame timestamp */
  timestamp: number;
  
  /** Waveform data (time domain) */
  waveform?: Float32Array;
  
  /** Spectrum data (frequency domain) */
  spectrum?: SpectralData;
  
  /** Peak level in dB */
  peak_level_db?: number;
  
  /** RMS level in dB */
  rms_level_db?: number;
  
  /** Gate state */
  gate_open?: boolean;
  
  /** Noise profile if available */
  noise_profile?: NoiseProfile;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Deep partial type for configuration updates
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Result type for async operations
 */
export type Result<T, E = ProcessingError> = 
  | { success: true; value: T }
  | { success: false; error: E };

/**
 * Disposable resource interface
 */
export interface Disposable {
  dispose(): void | Promise<void>;
}

/**
 * Event emitter interface for processors
 */
export interface ProcessorEventMap {
  'statechange': { state: 'running' | 'suspended' | 'closed' };
  'error': ProcessingError;
  'stats': ProcessingStats;
  'noisedetected': { level: number; timestamp: number };
  'gatechange': { open: boolean; timestamp: number };
  'overload': { cpu: number; timestamp: number };
}

/**
 * Type-safe event listener
 */
export type ProcessorEventListener<K extends keyof ProcessorEventMap> = (
  event: ProcessorEventMap[K]
) => void;

// ============================================================================
// Export Convenience Types
// ============================================================================

/**
 * Complete processor options combining all configurations
 */
export type CompleteProcessorOptions = ProcessorConfig & WasmInitOptions;

/**
 * Simplified API for common use cases
 */
export interface SimpleNoiseReducer {
  /** Remove noise with automatic settings */
  removeNoise(audioBuffer: AudioBuffer): Promise<AudioBuffer>;
  
  /** Learn noise profile from sample */
  learnNoise(noiseSample: AudioBuffer): Promise<void>;
  
  /** Reset to defaults */
  reset(): void;
}

/**
 * Presets for common scenarios
 */
export const PRESETS = {
  VOICE_CALL: {
    sample_rate: 48000 as SampleRate,
    frame_size: 256 as FrameSize,
    noise_gate: {
      threshold_db: -40,
      attack_ms: 5,
      release_ms: 100,
    },
    spectral_subtraction: {
      fft_size: 512 as FrameSize,
      alpha: 2.0,
      beta: 0.1,
    },
  },
  MUSIC_PRODUCTION: {
    sample_rate: 96000 as SampleRate,
    frame_size: 512 as FrameSize,
    noise_gate: {
      threshold_db: -60,
      attack_ms: 0.1,
      release_ms: 50,
      soft_knee: true,
    },
    spectral_subtraction: {
      fft_size: 2048 as FrameSize,
      alpha: 1.5,
      beta: 0.05,
    },
  },
  PODCAST: {
    sample_rate: 48000 as SampleRate,
    frame_size: 512 as FrameSize,
    noise_gate: {
      threshold_db: -45,
      attack_ms: 10,
      release_ms: 200,
    },
    spectral_subtraction: {
      fft_size: 1024 as FrameSize,
      alpha: 2.5,
      beta: 0.15,
    },
  },
} as const;

/**
 * Type guard for checking if value is ProcessingError
 */
export function isProcessingError(value: any): value is ProcessingError {
  return value &&
    typeof value === 'object' &&
    'type' in value &&
    'message' in value &&
    'timestamp' in value;
}

/**
 * Type guard for checking valid sample rate
 */
export function isValidSampleRate(rate: number): rate is SampleRate {
  return [44100, 48000, 88200, 96000].includes(rate);
}

/**
 * Type guard for checking valid frame size
 */
export function isValidFrameSize(size: number): size is FrameSize {
  return [128, 256, 512, 1024, 2048, 4096].includes(size) &&
    (size & (size - 1)) === 0; // Check power of 2
}
