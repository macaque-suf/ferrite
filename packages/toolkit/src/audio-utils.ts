/**
 * Audio utility functions for Web Audio API and WASM integration
 * 
 * This module provides browser-specific audio utilities and acts as a bridge
 * between the Web Audio API and our Rust/WASM processing core.
 * Heavy DSP processing is delegated to the WASM module.
 */

import type { 
  ProcessorConfig, 
  AudioFormat, 
  BufferStats,
  SpectralData,
  ProcessingError,
  ProcessingErrorType 
} from './types';

// ============================================================================
// Constants
// ============================================================================

/** Default sample rate if none specified */
export const DEFAULT_SAMPLE_RATE = 48000;

/** Default FFT size for spectral processing */
export const DEFAULT_FFT_SIZE = 512;

// ============================================================================
// Audio Context Management
// ============================================================================

let globalAudioContext: AudioContext | null = null;

/**
 * Get or create the global audio context
 * Handles user gesture requirements for Chrome/Safari
 */
export function getAudioContext(): AudioContext {
  if (!globalAudioContext || globalAudioContext.state === 'closed') {
    globalAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Resume context if suspended (Chrome autoplay policy)
    if (globalAudioContext.state === 'suspended') {
      globalAudioContext.resume();
    }
  }
  return globalAudioContext;
}

/**
 * Resume audio context if suspended (requires user interaction)
 */
export async function resumeAudioContext(): Promise<void> {
  const ctx = getAudioContext();
  if (ctx.state === 'suspended') {
    await ctx.resume();
  }
}

/**
 * Create an offline audio context for non-realtime processing
 */
export function createOfflineContext(
  channels: number,
  length: number,
  sampleRate: number
): OfflineAudioContext {
  return new OfflineAudioContext(channels, length, sampleRate);
}

// ============================================================================
// Web Audio Buffer Operations (Browser-specific)
// ============================================================================

/**
 * Create an AudioBuffer from Float32Array data
 * This is browser-specific and can't be done in WASM
 */
export function createAudioBuffer(
  data: Float32Array | number[],
  sampleRate: number = DEFAULT_SAMPLE_RATE,
  channels: number = 1
): AudioBuffer {
  const ctx = getAudioContext();
  const frameLength = Math.floor(data.length / channels);
  const buffer = ctx.createBuffer(channels, frameLength, sampleRate);
  
  const floatData = data instanceof Float32Array ? data : new Float32Array(data);
  
  for (let channel = 0; channel < channels; channel++) {
    const channelData = buffer.getChannelData(channel);
    if (channels === 1) {
      channelData.set(floatData);
    } else {
      // Deinterleave for multi-channel
      for (let i = 0; i < frameLength; i++) {
        channelData[i] = floatData[i * channels + channel];
      }
    }
  }
  
  return buffer;
}

/**
 * Extract Float32Array from AudioBuffer for WASM processing
 */
export function extractChannelData(
  buffer: AudioBuffer,
  channel: number = 0
): Float32Array {
  return buffer.getChannelData(Math.min(channel, buffer.numberOfChannels - 1));
}

/**
 * Extract and interleave all channels for WASM processing
 */
export function extractInterleavedData(buffer: AudioBuffer): Float32Array {
  const channels = buffer.numberOfChannels;
  const length = buffer.length;
  const interleaved = new Float32Array(length * channels);
  
  for (let i = 0; i < length; i++) {
    for (let ch = 0; ch < channels; ch++) {
      interleaved[i * channels + ch] = buffer.getChannelData(ch)[i];
    }
  }
  
  return interleaved;
}

/**
 * Clone an AudioBuffer
 */
export function cloneAudioBuffer(source: AudioBuffer): AudioBuffer {
  const ctx = getAudioContext();
  const clone = ctx.createBuffer(
    source.numberOfChannels,
    source.length,
    source.sampleRate
  );
  
  for (let channel = 0; channel < source.numberOfChannels; channel++) {
    clone.copyToChannel(source.getChannelData(channel), channel);
  }
  
  return clone;
}

// ============================================================================
// Media Stream Handling (Browser-specific)
// ============================================================================

/**
 * Get user audio stream with proper constraints
 */
export async function getUserAudioStream(
  constraints: MediaStreamConstraints = {}
): Promise<MediaStream> {
  const audioConstraints = constraints.audio || {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    sampleRate: DEFAULT_SAMPLE_RATE
  };
  
  return navigator.mediaDevices.getUserMedia({
    ...constraints,
    audio: audioConstraints,
    video: false
  });
}

/**
 * Create a MediaStreamSource node
 */
export function createStreamSource(
  stream: MediaStream,
  context?: AudioContext
): MediaStreamAudioSourceNode {
  const ctx = context || getAudioContext();
  return ctx.createMediaStreamSource(stream);
}

/**
 * Create a ScriptProcessor node for real-time processing
 * Note: ScriptProcessor is deprecated but more widely supported than AudioWorklet
 */
export function createProcessor(
  inputChannels: number = 1,
  outputChannels: number = 1,
  bufferSize: number = 512,
  context?: AudioContext
): ScriptProcessorNode {
  const ctx = context || getAudioContext();
  return ctx.createScriptProcessor(bufferSize, inputChannels, outputChannels);
}

/**
 * Stop all tracks in a media stream
 */
export function stopStream(stream: MediaStream): void {
  stream.getTracks().forEach(track => track.stop());
}

// ============================================================================
// AudioWorklet Support (Modern Alternative)
// ============================================================================

/**
 * Check if AudioWorklet is supported
 */
export function isAudioWorkletSupported(): boolean {
  return typeof AudioWorkletNode !== 'undefined';
}

/**
 * Load AudioWorklet processor module
 */
export async function loadAudioWorklet(
  moduleUrl: string,
  context?: AudioContext
): Promise<void> {
  const ctx = context || getAudioContext();
  if (!ctx.audioWorklet) {
    throw new Error('AudioWorklet not supported in this browser');
  }
  await ctx.audioWorklet.addModule(moduleUrl);
}

/**
 * Create AudioWorklet node
 */
export function createAudioWorkletNode(
  processorName: string,
  options?: AudioWorkletNodeOptions,
  context?: AudioContext
): AudioWorkletNode {
  const ctx = context || getAudioContext();
  return new AudioWorkletNode(ctx, processorName, options);
}

// ============================================================================
// File I/O (Browser-specific)
// ============================================================================

/**
 * Load audio file as AudioBuffer
 */
export async function loadAudioFile(file: File | Blob): Promise<AudioBuffer> {
  const ctx = getAudioContext();
  const arrayBuffer = await file.arrayBuffer();
  return ctx.decodeAudioData(arrayBuffer);
}

/**
 * Load audio from URL
 */
export async function loadAudioFromURL(url: string): Promise<AudioBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load audio from ${url}: ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const ctx = getAudioContext();
  return ctx.decodeAudioData(arrayBuffer);
}

/**
 * Convert AudioBuffer to WAV blob for download
 * This is browser-specific functionality
 */
export function audioBufferToWav(buffer: AudioBuffer): Blob {
  const length = buffer.length * buffer.numberOfChannels * 2;
  const arrayBuffer = new ArrayBuffer(44 + length);
  const view = new DataView(arrayBuffer);
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, buffer.numberOfChannels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * buffer.numberOfChannels * 2, true);
  view.setUint16(32, buffer.numberOfChannels * 2, true);
  view.setUint16(34, 16, true); // 16-bit samples
  writeString(36, 'data');
  view.setUint32(40, length, true);
  
  // Interleave and convert to 16-bit PCM
  let offset = 44;
  const channels: Float32Array[] = [];
  for (let i = 0; i < buffer.numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }
  
  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }
  }
  
  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

/**
 * Download AudioBuffer as WAV file
 */
export function downloadAudioBuffer(
  buffer: AudioBuffer,
  filename: string = 'audio.wav'
): void {
  const blob = audioBufferToWav(buffer);
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  
  URL.revokeObjectURL(url);
}

// ============================================================================
// WASM Integration Helpers
// ============================================================================

/**
 * Prepare audio data for WASM processing
 * Ensures data is in the correct format and within valid ranges
 */
export function prepareForWasm(data: Float32Array): Float32Array {
  // Ensure no NaN or Infinity values
  const cleaned = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const sample = data[i];
    if (!isFinite(sample)) {
      cleaned[i] = 0;
    } else {
      // Clamp to valid range
      cleaned[i] = Math.max(-1, Math.min(1, sample));
    }
  }
  return cleaned;
}

/**
 * Create shared memory for WASM if supported
 */
export function createSharedBuffer(size: number): SharedArrayBuffer | ArrayBuffer {
  if (typeof SharedArrayBuffer !== 'undefined') {
    try {
      return new SharedArrayBuffer(size * Float32Array.BYTES_PER_ELEMENT);
    } catch (e) {
      // SharedArrayBuffer may be disabled due to Spectre mitigations
    }
  }
  return new ArrayBuffer(size * Float32Array.BYTES_PER_ELEMENT);
}

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Create a processing error
 */
export function createProcessingError(
  type: ProcessingErrorType,
  message: string,
  details?: Record<string, any>
): ProcessingError {
  return {
    type,
    message,
    details,
    timestamp: Date.now(),
    recoverable: type !== ProcessingErrorType.INITIALIZATION_FAILED,
    stack: new Error().stack
  };
}

// ============================================================================
// Browser Capability Detection
// ============================================================================

/**
 * Detect browser audio capabilities
 */
export function detectCapabilities(): {
  audioContext: boolean;
  getUserMedia: boolean;
  audioWorklet: boolean;
  sharedArrayBuffer: boolean;
  offlineContext: boolean;
  mediaRecorder: boolean;
} {
  return {
    audioContext: !!(window.AudioContext || (window as any).webkitAudioContext),
    getUserMedia: !!(navigator.mediaDevices?.getUserMedia),
    audioWorklet: typeof AudioWorkletNode !== 'undefined',
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    offlineContext: typeof OfflineAudioContext !== 'undefined',
    mediaRecorder: typeof MediaRecorder !== 'undefined'
  };
}

/**
 * Check if running in a secure context (required for some APIs)
 */
export function isSecureContext(): boolean {
  return window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
}

// ============================================================================
// Utility Functions (Browser-specific)
// ============================================================================

/**
 * Format time in seconds to MM:SS
 */
export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Get audio buffer info as string (for debugging)
 */
export function bufferToString(buffer: AudioBuffer): string {
  return `AudioBuffer[channels=${buffer.numberOfChannels}, length=${buffer.length}, rate=${buffer.sampleRate}Hz, duration=${buffer.duration.toFixed(2)}s]`;
}

/**
 * Estimate processing latency based on buffer size and sample rate
 */
export function estimateLatency(bufferSize: number, sampleRate: number): number {
  return (bufferSize / sampleRate) * 1000; // in milliseconds
}

// ============================================================================
// Web Worker Support
// ============================================================================

/**
 * Check if Web Workers are supported
 */
export function isWorkerSupported(): boolean {
  return typeof Worker !== 'undefined';
}

/**
 * Create a processing worker
 */
export function createProcessingWorker(workerUrl: string): Worker | null {
  if (!isWorkerSupported()) {
    return null;
  }
  return new Worker(workerUrl, { type: 'module' });
}

/**
 * Transfer audio data to worker efficiently
 */
export function transferToWorker(
  worker: Worker,
  data: Float32Array,
  transfer: boolean = true
): void {
  if (transfer && data.buffer instanceof ArrayBuffer) {
    // Transfer ownership for zero-copy
    worker.postMessage({ audio: data }, [data.buffer]);
  } else {
    // Regular copy
    worker.postMessage({ audio: data });
  }
}

// ============================================================================
// Export namespace for convenience
// ============================================================================

export const AudioUtils = {
  // Context management
  getAudioContext,
  resumeAudioContext,
  createOfflineContext,
  
  // Buffer operations
  createAudioBuffer,
  extractChannelData,
  extractInterleavedData,
  cloneAudioBuffer,
  
  // Streaming
  getUserAudioStream,
  createStreamSource,
  createProcessor,
  stopStream,
  
  // AudioWorklet
  isAudioWorkletSupported,
  loadAudioWorklet,
  createAudioWorkletNode,
  
  // File I/O
  loadAudioFile,
  loadAudioFromURL,
  audioBufferToWav,
  downloadAudioBuffer,
  
  // WASM integration
  prepareForWasm,
  createSharedBuffer,
  
  // Error handling
  createProcessingError,
  
  // Capabilities
  detectCapabilities,
  isSecureContext,
  
  // Workers
  isWorkerSupported,
  createProcessingWorker,
  transferToWorker,
  
  // Utilities
  formatTime,
  bufferToString,
  estimateLatency,
};

export default AudioUtils;
