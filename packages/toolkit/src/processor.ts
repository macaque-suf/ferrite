/**
 * TypeScript wrapper for WASM audio processing module
 * Provides clean API for Web Audio integration and real-time processing
 */

import init, { NoiseReducer as WasmNoiseReducer } from '../../../dist/wasm/wasm_audio_ferrite';
import { 
  NoiseProfile,
  ProcessingStats as StatsType,
  ProcessingError,
  ProcessingErrorType
} from './types';
import { createAudioBuffer, getAudioContext } from './audio-utils';

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface ProcessorOptions {
  sampleRate?: number;
  fftSize?: number;
  frameSize?: number;
  workerMode?: boolean;
  workletMode?: boolean;
  latencyHint?: 'interactive' | 'balanced' | 'playback';
  
  // Noise gate settings
  gateEnabled?: boolean;
  gateThreshold?: number;
  gateRatio?: number;
  gateAttack?: number;
  gateRelease?: number;
  
  // Spectral subtraction settings
  spectralEnabled?: boolean;
  overSubtractionFactor?: number;
  spectralFloor?: number;
  noiseEstimationMode?: 'adaptive' | 'manual';
}

export interface ProcessorState {
  isProcessing: boolean;
  gateState: 'open' | 'closed' | 'opening' | 'closing';
  currentGainReduction: number;
  noiseProfileLocked: boolean;
  latencySamples: number;
  cpuUsage: number;
}

export interface ProcessingStats {
  samplesProcessed: number;
  framesProcessed: number;
  averageLatencyMs: number;
  peakLatencyMs: number;
  gateOpenTime: number;
  gateClosedTime: number;
  cpuUsagePercent: number;
  droppedFrames: number;
}

export interface BrowserSupport {
  audioWorklet: boolean;
  webAssembly: boolean;
  sharedArrayBuffer: boolean;
  webAudioApi: boolean;
  mediaDevices: boolean;
}

export interface MemoryStats {
  used: number;
  total: number;
  peak: number;
}

export interface SimdSupport {
  available: boolean;
  version: string;
  features: string[];
}

export enum NoiseReductionPreset {
  VoiceChat = 'voice-chat',
  MusicRecording = 'music-recording',
  PodcastRecording = 'podcast',
  Aggressive = 'aggressive',
  Gentle = 'gentle',
  Custom = 'custom'
}

export enum ProcessorErrorCode {
  WasmNotInitialized = 'WASM_NOT_INITIALIZED',
  AudioContextSuspended = 'AUDIO_CONTEXT_SUSPENDED',
  WorkletNotSupported = 'WORKLET_NOT_SUPPORTED',
  InvalidConfiguration = 'INVALID_CONFIGURATION',
  BufferSizeMismatch = 'BUFFER_SIZE_MISMATCH',
  ProcessingTimeout = 'PROCESSING_TIMEOUT'
}

export class ProcessorError extends Error {
  code: ProcessorErrorCode;
  details?: any;
  timestamp: number;
  recoverable: boolean;
  type: ProcessingErrorType;
  
  constructor(code: ProcessorErrorCode, message: string, details?: any) {
    super(message);
    this.name = 'ProcessorError';
    this.code = code;
    this.details = details;
    this.timestamp = Date.now();
    this.recoverable = false;
    this.type = ProcessingErrorType.UNKNOWN;
  }
}

// ============================================================================
// Buffer Pool for Performance
// ============================================================================

export class BufferPool {
  private pools: Map<number, Float32Array[]> = new Map();
  private maxPoolSize = 10;
  
  acquire(size: number): Float32Array {
    const pool = this.pools.get(size) || [];
    
    if (pool.length > 0) {
      return pool.pop()!;
    }
    
    return new Float32Array(size);
  }
  
  release(buffer: Float32Array): void {
    const size = buffer.length;
    
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    
    const pool = this.pools.get(size)!;
    
    if (pool.length < this.maxPoolSize) {
      buffer.fill(0); // Clear the buffer
      pool.push(buffer);
    }
  }
  
  clear(): void {
    this.pools.clear();
  }
}

// ============================================================================
// Main NoiseReducer Class
// ============================================================================

export class NoiseReducer {
  private wasmModule: WasmNoiseReducer | null = null;
  private wasmInitialized = false;
  private wasmMemory: WebAssembly.Memory | null = null;
  private wasmInputPtr: number = 0;
  private wasmOutputPtr: number = 0;
  private wasmBufferSize: number = 0;
  private config: Required<ProcessorOptions>;
  private bufferPool: BufferPool;
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private errorHandler: ((error: ProcessorError) => void) | null = null;
  
  // State tracking
  private state: ProcessorState = {
    isProcessing: false,
    gateState: 'closed',
    currentGainReduction: 0,
    noiseProfileLocked: false,
    latencySamples: 0,
    cpuUsage: 0
  };
  
  // Statistics tracking
  private stats: ProcessingStats = {
    samplesProcessed: 0,
    framesProcessed: 0,
    averageLatencyMs: 0,
    peakLatencyMs: 0,
    gateOpenTime: 0,
    gateClosedTime: 0,
    cpuUsagePercent: 0,
    droppedFrames: 0
  };
  
  // Noise profile management
  private noiseProfile: Float32Array | null = null;
  private isCapturingNoise = false;
  private noiseSamples: Float32Array[] = [];
  
  // Performance monitoring
  private perfStartTime = 0;
  private perfSamples: number[] = [];
  
  constructor(options: ProcessorOptions = {}) {
    // Set default configuration
    this.config = {
      sampleRate: options.sampleRate || 48000,
      fftSize: options.fftSize || 512,
      frameSize: options.frameSize || 256,
      workerMode: options.workerMode || false,
      workletMode: options.workletMode !== false, // Default true
      latencyHint: options.latencyHint || 'interactive',
      
      // Noise gate defaults
      gateEnabled: options.gateEnabled !== false,
      gateThreshold: options.gateThreshold || -40,
      gateRatio: options.gateRatio || 10,
      gateAttack: options.gateAttack || 5,
      gateRelease: options.gateRelease || 100,
      
      // Spectral defaults
      spectralEnabled: options.spectralEnabled !== false,
      overSubtractionFactor: options.overSubtractionFactor || 2.0,
      spectralFloor: options.spectralFloor || 0.1,
      noiseEstimationMode: options.noiseEstimationMode || 'adaptive'
    };
    
    this.bufferPool = new BufferPool();
  }
  
  // ============================================================================
  // Initialization and Lifecycle
  // ============================================================================
  
  async initialize(): Promise<void> {
    if (this.wasmInitialized) return;
    
    try {
      // Initialize WASM module
      await init();
      this.wasmModule = new WasmNoiseReducer(this.config.sampleRate);
      console.log('[Processor] WASM module created:', this.wasmModule);
      console.log('[Processor] Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(this.wasmModule)));
      this.wasmInitialized = true;
      
      // Get or create audio context
      this.audioContext = await getAudioContext();
      
      // Configure initial parameters
      this.applyConfiguration();
      
    } catch (error) {
      this.handleError(new ProcessorError(
        ProcessorErrorCode.WasmNotInitialized,
        'Failed to initialize WASM module',
        error
      ));
      throw error;
    }
  }
  
  isInitialized(): boolean {
    return this.wasmInitialized;
  }
  
  dispose(): void {
    // Clean up audio nodes
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    
    if (this.scriptNode) {
      this.scriptNode.disconnect();
      this.scriptNode = null;
    }
    
    // Clean up WASM module
    if (this.wasmModule) {
      this.wasmModule.free();
      this.wasmModule = null;
    }
    
    // Clear buffer pool
    this.bufferPool.clear();
    
    // Reset state
    this.wasmInitialized = false;
    this.state.isProcessing = false;
  }
  
  // ============================================================================
  // Audio Processing Methods
  // ============================================================================
  
  async process(audioBuffer: AudioBuffer): Promise<AudioBuffer> {
    if (!this.wasmInitialized) {
      await this.initialize();
    }
    
    const startTime = performance.now();
    
    try {
      // Get input data
      const inputData = audioBuffer.getChannelData(0);
      
      // Process through WASM
      const processed = await this.processFloat32Array(inputData);
      
      // Create output buffer
      const outputBuffer = createAudioBuffer(
        processed,
        this.config.sampleRate,
        1
      );
      
      // Update statistics
      this.updateStats(performance.now() - startTime, processed.length);
      
      return outputBuffer;
      
    } catch (error) {
      this.handleError(new ProcessorError(
        ProcessorErrorCode.ProcessingTimeout,
        'Processing failed',
        error
      ));
      throw error;
    }
  }
  
  async processFloat32Array(input: Float32Array): Promise<Float32Array> {
    if (!this.wasmModule) {
      throw new ProcessorError(
        ProcessorErrorCode.WasmNotInitialized,
        'WASM module not initialized'
      );
    }
    
    const startTime = performance.now();
    
    // If capturing noise, store the samples for profile learning
    if (this.isCapturingNoise) {
      // Store a copy of the input for noise profile analysis
      this.noiseSamples.push(new Float32Array(input));
      console.log('[Processor] Captured noise sample, total samples:', this.noiseSamples.length);
    }
    
    // Get buffer from pool for efficiency
    const output = this.bufferPool.acquire(input.length);
    
    try {
      // Process in chunks if needed
      const chunkSize = this.config.frameSize;
      
      for (let i = 0; i < input.length; i += chunkSize) {
        const end = Math.min(i + chunkSize, input.length);
        const chunk = input.subarray(i, end);
        const processed = this.wasmModule.process(chunk);
        output.set(new Float32Array(processed), i);
      }
      
      // Update performance statistics
      this.updateStats(performance.now() - startTime, input.length);
      
      return output;
      
    } catch (error) {
      this.bufferPool.release(output);
      throw error;
    }
  }
  
  processInPlace(buffer: Float32Array): void {
    if (!this.wasmModule) {
      throw new ProcessorError(
        ProcessorErrorCode.WasmNotInitialized,
        'WASM module not initialized'
      );
    }
    
    const processed = this.wasmModule.process(buffer);
    buffer.set(new Float32Array(processed));
  }
  
  // ============================================================================
  // Zero-Copy Processing API for Real-Time Performance
  // ============================================================================
  
  /**
   * Process audio frame with zero-copy optimization.
   * Uses pre-allocated WASM buffers to avoid allocation and copying.
   * 
   * @param input - Input audio frame
   * @param output - Optional output buffer (if not provided, processes in-place)
   * @returns Processed audio frame
   */
  processFrame(input: Float32Array, output?: Float32Array): Float32Array {
    if (!this.wasmModule || !this.wasmMemory) {
      // Fall back to regular processing if zero-copy not available
      if (this.wasmModule) {
        const result = this.wasmModule.process(input);
        const processed = new Float32Array(result);
        if (output) {
          output.set(processed);
          return output;
        }
        return processed;
      }
      throw new ProcessorError(
        ProcessorErrorCode.WasmNotInitialized,
        'WASM module not initialized'
      );
    }
    
    const startTime = performance.now();
    const len = input.length;
    
    // Ensure we have allocated buffers of the right size
    if (this.wasmBufferSize !== len) {
      this.allocateWasmBuffers(len);
    }
    
    // Get views into WASM memory
    const wasmInput = new Float32Array(this.wasmMemory.buffer, this.wasmInputPtr, len);
    
    // Copy input to WASM memory
    wasmInput.set(input);
    
    if (output) {
      // Process from input buffer to output buffer
      const wasmOutput = new Float32Array(this.wasmMemory.buffer, this.wasmOutputPtr, len);
      (this.wasmModule as any).processInto(this.wasmInputPtr, this.wasmOutputPtr, len);
      output.set(wasmOutput);
      
      // Update statistics
      this.updateStats(performance.now() - startTime, len);
      return output;
    } else {
      // Process in-place
      (this.wasmModule as any).processPtr(this.wasmInputPtr, len);
      
      // Copy result back
      const result = new Float32Array(wasmInput);
      
      // Update statistics
      this.updateStats(performance.now() - startTime, len);
      return result;
    }
  }
  
  /**
   * Allocate buffers in WASM memory for zero-copy processing
   */
  private allocateWasmBuffers(size: number): void {
    if (!this.wasmModule) return;
    
    // Check if the module has the zero-copy methods
    if (!(this.wasmModule as any).alloc_buffer) {
      console.warn('[Processor] Zero-copy API not available in WASM module');
      return;
    }
    
    // Allocate input and output buffers in WASM memory
    this.wasmInputPtr = (this.wasmModule as any).alloc_buffer(size);
    this.wasmOutputPtr = (this.wasmModule as any).alloc_buffer(size);
    this.wasmBufferSize = size;
    
    // Get reference to WASM memory
    // Try to get from the module's exports
    const wasmExports = (this.wasmModule as any).__wbg_get_exports?.() || this.wasmModule;
    if (wasmExports.memory) {
      this.wasmMemory = wasmExports.memory;
    } else if ((window as any).__wasm_memory) {
      // Fallback to global if available
      this.wasmMemory = (window as any).__wasm_memory;
    }
    
    if (!this.wasmMemory) {
      console.warn('[Processor] Could not access WASM memory for zero-copy');
    }
  }
  
  static async removeNoise(audioBuffer: AudioBuffer): Promise<AudioBuffer> {
    const processor = new NoiseReducer({ 
      sampleRate: audioBuffer.sampleRate 
    });
    
    try {
      await processor.initialize();
      return await processor.process(audioBuffer);
    } finally {
      processor.dispose();
    }
  }
  
  // ============================================================================
  // Web Audio Integration
  // ============================================================================
  
  async registerWorklet(audioContext: AudioContext): Promise<void> {
    if (!audioContext.audioWorklet) {
      throw new ProcessorError(
        ProcessorErrorCode.WorkletNotSupported,
        'AudioWorklet not supported in this browser'
      );
    }
    
    // Register the worklet processor
    await audioContext.audioWorklet.addModule('/noise-reducer-worklet.js');
  }
  
  async createWorkletNode(audioContext: AudioContext): Promise<AudioWorkletNode> {
    if (!this.wasmInitialized) {
      await this.initialize();
    }
    
    await this.registerWorklet(audioContext);
    
    this.workletNode = new AudioWorkletNode(audioContext, 'noise-reducer-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: this.config
    });
    
    // Set up message handling
    this.workletNode.port.onmessage = (event) => {
      this.handleWorkletMessage(event.data);
    };
    
    return this.workletNode;
  }
  
  createScriptProcessor(audioContext: AudioContext): ScriptProcessorNode {
    const bufferSize = this.config.frameSize * 2; // Double buffer for stability
    
    this.scriptNode = audioContext.createScriptProcessor(
      bufferSize,
      1, // input channels
      1  // output channels
    );
    
    this.scriptNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const output = event.outputBuffer.getChannelData(0);
      
      if (this.wasmModule && this.state.isProcessing) {
        const processed = this.wasmModule.process(input);
        output.set(new Float32Array(processed));
      } else {
        output.set(input); // Pass through
      }
    };
    
    return this.scriptNode;
  }
  
  createProcessor(): AudioWorkletNode | ScriptProcessorNode {
    if (!this.audioContext) {
      throw new ProcessorError(
        ProcessorErrorCode.AudioContextSuspended,
        'Audio context not initialized'
      );
    }
    
    if (this.config.workletMode && this.audioContext.audioWorklet) {
      return this.createWorkletNode(this.audioContext) as any;
    } else {
      return this.createScriptProcessor(this.audioContext);
    }
  }
  
  connectToStream(sourceNode: AudioNode): AudioNode {
    const processor = this.createProcessor();
    sourceNode.connect(processor);
    return processor;
  }
  
  async processMediaStream(stream: MediaStream): Promise<MediaStream> {
    const audioContext = await getAudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const destination = audioContext.createMediaStreamDestination();
    
    const processor = this.connectToStream(source);
    processor.connect(destination);
    
    return destination.stream;
  }
  
  // ============================================================================
  // Configuration Management
  // ============================================================================
  
  setGateThreshold(dbValue: number): void {
    this.config.gateThreshold = dbValue;
    // The actual WASM doesn't have a gate threshold method
    // It uses reduction_amount to control the overall effect
    // Map threshold to reduction amount
    if (this.wasmModule && typeof this.wasmModule.set_reduction_amount === 'function') {
      // Convert threshold to a reduction amount
      // Higher threshold = less reduction
      const normalized = (dbValue + 80) / 80; // Normalize -80 to 0 => 0 to 1
      const amount = Math.max(0, Math.min(1, 1 - normalized));
      this.wasmModule.set_reduction_amount(amount * this.config.overSubtractionFactor / 3.0);
    }
  }
  
  setGateEnabled(enabled: boolean): void {
    this.config.gateEnabled = enabled;
    // The actual WASM uses set_bypass for enabling/disabling processing
    if (this.wasmModule && typeof this.wasmModule.set_bypass === 'function') {
      this.wasmModule.set_bypass(!enabled); // bypass is opposite of enabled
    }
  }
  
  setSpectralEnabled(enabled: boolean): void {
    this.config.spectralEnabled = enabled;
    // The actual WASM combines this with reduction_amount
    if (this.wasmModule && typeof this.wasmModule.set_reduction_amount === 'function') {
      const amount = enabled ? this.config.overSubtractionFactor / 3.0 : 0;
      this.wasmModule.set_reduction_amount(amount);
    }
  }
  
  setOverSubtractionFactor(factor: number): void {
    this.config.overSubtractionFactor = factor;
    // Map to reduction_amount (0.0 to 1.0)
    if (this.wasmModule && typeof this.wasmModule.set_reduction_amount === 'function') {
      const amount = Math.min(1.0, factor / 3.0); // normalize to 0-1
      this.wasmModule.set_reduction_amount(amount);
    }
  }
  
  setSpectralFloor(floor: number): void {
    this.config.spectralFloor = floor;
    // The actual WASM doesn't have this specific method, it's part of reduction_amount
    this.updateReductionAmount();
  }
  
  private updateReductionAmount(): void {
    if (this.wasmModule && typeof this.wasmModule.set_reduction_amount === 'function') {
      // Combine various factors into a single reduction amount
      const factor = this.config.spectralEnabled ? this.config.overSubtractionFactor : 0;
      const amount = Math.min(1.0, factor / 3.0);
      this.wasmModule.set_reduction_amount(amount);
    }
  }
  
  updateConfig(config: Partial<ProcessorOptions>): void {
    Object.assign(this.config, config);
    this.applyConfiguration();
  }
  
  getConfig(): ProcessorOptions {
    return { ...this.config };
  }
  
  loadPreset(preset: NoiseReductionPreset): void {
    const presetConfig = getPresetConfig(preset);
    this.updateConfig(presetConfig);
  }
  
  savePreset(): NoiseReductionPreset {
    // Save current config as custom preset
    return NoiseReductionPreset.Custom;
  }
  
  private applyConfiguration(): void {
    if (!this.wasmModule) return;
    
    // Apply configuration using the actual WASM API
    try {
      // The actual WASM uses set_bypass and set_reduction_amount
      if (typeof this.wasmModule.set_bypass === 'function') {
        this.wasmModule.set_bypass(!this.config.gateEnabled);
      }
      
      if (typeof this.wasmModule.set_reduction_amount === 'function') {
        // Map our configuration to reduction amount (0.0 to 1.0)
        const amount = this.config.spectralEnabled 
          ? Math.min(1.0, this.config.overSubtractionFactor / 3.0)
          : 0;
        this.wasmModule.set_reduction_amount(amount);
      }
      
      console.log('[Processor] Configuration applied - bypass:', !this.config.gateEnabled, 
                  'reduction:', this.config.overSubtractionFactor / 3.0);
    } catch (error) {
      console.warn('Error applying WASM configuration:', error);
    }
  }
  
  // ============================================================================
  // Noise Profile Management
  // ============================================================================
  
  startNoiseCapture(): void {
    this.isCapturingNoise = true;
    this.noiseSamples = [];
  }
  
  stopNoiseCapture(): NoiseProfile {
    this.isCapturingNoise = false;
    
    // Average captured samples
    if (this.noiseSamples.length > 0) {
      const avgLength = this.noiseSamples[0].length;
      const averaged = new Float32Array(avgLength);
      
      for (let i = 0; i < avgLength; i++) {
        let sum = 0;
        for (const sample of this.noiseSamples) {
          sum += sample[i];
        }
        averaged[i] = sum / this.noiseSamples.length;
      }
      
      this.noiseProfile = averaged;
      
      // Send the learned noise profile to the WASM module
      if (this.wasmModule && typeof this.wasmModule.learn_noise === 'function') {
        console.log('[Processor] Learning noise profile with', this.noiseSamples.length, 'samples');
        this.wasmModule.learn_noise(averaged);
      }
    }
    
    return {
      spectrum: this.noiseProfile || new Float32Array(0),
      frame_count: this.noiseSamples.length,
      timestamp: Date.now(),
      is_locked: true
    };
  }
  
  learnNoiseProfile(samples: Float32Array, duration?: number): void {
    if (this.wasmModule && typeof this.wasmModule.learn_noise === 'function') {
      // The actual WASM uses learn_noise
      this.wasmModule.learn_noise(samples);
    }
    
    this.noiseProfile = samples;
    this.state.noiseProfileLocked = true;
  }
  
  setNoiseProfile(profile: NoiseProfile): void {
    this.noiseProfile = profile.spectrum;
    
    // The actual WASM uses learn_noise for setting profiles
    if (this.wasmModule && typeof this.wasmModule.learn_noise === 'function') {
      this.wasmModule.learn_noise(profile.spectrum);
    }
  }
  
  getNoiseProfile(): NoiseProfile {
    return {
      spectrum: this.noiseProfile || new Float32Array(0),
      frame_count: 0,
      timestamp: Date.now(),
      is_locked: this.state.noiseProfileLocked
    };
  }
  
  freezeNoiseProfile(): void {
    this.state.noiseProfileLocked = true;
    // The actual WASM doesn't have a freeze method, it's managed in JS
  }
  
  unfreezeNoiseProfile(): void {
    this.state.noiseProfileLocked = false;
    // The actual WASM doesn't have an unfreeze method, it's managed in JS
  }
  
  resetNoiseProfile(): void {
    this.noiseProfile = null;
    this.state.noiseProfileLocked = false;
    
    // The actual WASM has a reset method
    if (this.wasmModule && typeof this.wasmModule.reset === 'function') {
      this.wasmModule.reset();
    }
  }
  
  exportNoiseProfile(): string {
    const profile = this.getNoiseProfile();
    return JSON.stringify(profile);
  }
  
  importNoiseProfile(json: string): void {
    const profile = JSON.parse(json) as NoiseProfile;
    this.setNoiseProfile(profile);
  }
  
  // ============================================================================
  // State & Statistics
  // ============================================================================
  
  getState(): ProcessorState {
    return { ...this.state };
  }
  
  getStatistics(): ProcessingStats {
    return { ...this.stats };
  }
  
  reset(): void {
    // Reset state
    this.state = {
      isProcessing: false,
      gateState: 'closed',
      currentGainReduction: 0,
      noiseProfileLocked: false,
      latencySamples: 0,
      cpuUsage: 0
    };
    
    // Reset statistics
    this.stats = {
      samplesProcessed: 0,
      framesProcessed: 0,
      averageLatencyMs: 0,
      peakLatencyMs: 0,
      gateOpenTime: 0,
      gateClosedTime: 0,
      cpuUsagePercent: 0,
      droppedFrames: 0
    };
    
    // Reset WASM module
    if (this.wasmModule) {
      this.wasmModule.reset();
    }
  }
  
  private updateStats(processingTime: number, samples: number): void {
    this.stats.samplesProcessed += samples;
    this.stats.framesProcessed++;
    
    // Update latency stats
    const latencyMs = processingTime;
    this.stats.peakLatencyMs = Math.max(this.stats.peakLatencyMs, latencyMs);
    
    // Running average
    this.stats.averageLatencyMs = 
      (this.stats.averageLatencyMs * (this.stats.framesProcessed - 1) + latencyMs) / 
      this.stats.framesProcessed;
    
    // Estimate CPU usage
    const expectedTime = (samples / this.config.sampleRate) * 1000;
    this.stats.cpuUsagePercent = (processingTime / expectedTime) * 100;
  }
  
  // ============================================================================
  // Error Handling
  // ============================================================================
  
  handleError(error: ProcessorError): void {
    if (this.errorHandler) {
      this.errorHandler(error);
    } else {
      console.error(`ProcessorError [${error.code}]:`, error.message, error.details);
    }
  }
  
  setErrorHandler(handler: (error: ProcessorError) => void): void {
    this.errorHandler = handler;
  }
  
  // ============================================================================
  // Memory Management
  // ============================================================================
  
  allocateWasmMemory(size: number): number {
    if (!this.wasmModule) {
      throw new ProcessorError(
        ProcessorErrorCode.WasmNotInitialized,
        'WASM module not initialized'
      );
    }
    
    // The actual WASM doesn't expose memory allocation directly
    // Return a mock pointer for compatibility
    return Math.floor(Math.random() * 0xFFFFFF);
  }
  
  freeWasmMemory(ptr: number): void {
    // The actual WASM handles memory internally
    // This is a no-op for compatibility
  }
  
  getWasmMemoryUsage(): MemoryStats {
    // The actual WASM doesn't expose memory stats
    // Return estimates based on usage
    const estimatedUsage = this.stats.samplesProcessed * 4; // 4 bytes per float
    
    return {
      used: estimatedUsage,
      total: 16 * 1024 * 1024, // 16MB estimate
      peak: estimatedUsage * 1.5
    };
  }
  
  // ============================================================================
  // Performance Optimization
  // ============================================================================
  
  detectSimdSupport(): SimdSupport {
    const hasSimd = typeof WebAssembly !== 'undefined' && 
                   'validate' in WebAssembly &&
                   WebAssembly.validate(new Uint8Array([
                     0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                     0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
                     0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
                     0x41, 0x00, 0xfd, 0x0f, 0x0b, 0x00
                   ]));
    
    return {
      available: hasSimd,
      version: hasSimd ? 'wasm-simd' : 'none',
      features: hasSimd ? ['v128'] : []
    };
  }
  
  enableSimd(enabled: boolean): void {
    // The actual WASM doesn't have a SIMD toggle
    // SIMD is determined at compile time
    console.log('[Processor] SIMD request:', enabled, '(determined at WASM compile time)');
  }
  
  startPerfMonitoring(): void {
    this.perfStartTime = performance.now();
    this.perfSamples = [];
  }
  
  stopPerfMonitoring(): any {
    const duration = performance.now() - this.perfStartTime;
    
    return {
      duration,
      samples: this.perfSamples.length,
      avgProcessingTime: this.stats.averageLatencyMs,
      peakProcessingTime: this.stats.peakLatencyMs,
      cpuUsage: this.stats.cpuUsagePercent
    };
  }
  
  onPerformanceWarning(callback: (warning: any) => void): void {
    // Monitor for performance issues
    setInterval(() => {
      if (this.stats.cpuUsagePercent > 80) {
        callback({
          type: 'high-cpu',
          usage: this.stats.cpuUsagePercent,
          timestamp: Date.now()
        });
      }
      
      if (this.stats.droppedFrames > 0) {
        callback({
          type: 'dropped-frames',
          count: this.stats.droppedFrames,
          timestamp: Date.now()
        });
      }
    }, 1000);
  }
  
  // ============================================================================
  // Browser Compatibility
  // ============================================================================
  
  static checkBrowserSupport(): BrowserSupport {
    return {
      audioWorklet: 'AudioWorkletNode' in window,
      webAssembly: typeof WebAssembly !== 'undefined',
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      webAudioApi: 'AudioContext' in window || 'webkitAudioContext' in window,
      mediaDevices: navigator.mediaDevices && 
                    navigator.mediaDevices.getUserMedia !== undefined
    };
  }
  
  setFallbackMode(mode: 'script-processor' | 'web-worker' | 'main-thread'): void {
    switch (mode) {
      case 'script-processor':
        this.config.workletMode = false;
        this.config.workerMode = false;
        break;
      case 'web-worker':
        this.config.workerMode = true;
        this.config.workletMode = false;
        break;
      case 'main-thread':
        this.config.workerMode = false;
        this.config.workletMode = false;
        break;
    }
  }
  
  // ============================================================================
  // Private Helper Methods
  // ============================================================================
  
  private handleWorkletMessage(data: any): void {
    switch (data.type) {
      case 'state':
        this.state = data.state;
        break;
      case 'stats':
        this.stats = data.stats;
        break;
      case 'error':
        this.handleError(data.error);
        break;
    }
  }
}

// ============================================================================
// Integration Helpers
// ============================================================================

export class MicrophoneProcessor {
  private processor: NoiseReducer;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private audioContext: AudioContext | null = null;
  private onProcessedCallbacks: ((buffer: AudioBuffer) => void)[] = [];
  
  constructor(processor: NoiseReducer) {
    this.processor = processor;
  }
  
  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      } 
    });
    
    this.audioContext = await getAudioContext();
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    
    const processorNode = this.processor.connectToStream(this.source);
    
    // Set up recording if needed
    const recorder = this.audioContext.createScriptProcessor(4096, 1, 1);
    processorNode.connect(recorder);
    recorder.connect(this.audioContext.destination);
    
    recorder.onaudioprocess = (event) => {
      const buffer = event.inputBuffer;
      this.onProcessedCallbacks.forEach(callback => callback(buffer));
    };
  }
  
  stop(): void {
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }
  
  onProcessed(callback: (buffer: AudioBuffer) => void): void {
    this.onProcessedCallbacks.push(callback);
  }
  
  getNoiseReducer(): NoiseReducer {
    return this.processor;
  }
  
  static async create(
    constraints?: MediaStreamConstraints
  ): Promise<MicrophoneProcessor> {
    const processor = new NoiseReducer();
    await processor.initialize();
    return new MicrophoneProcessor(processor);
  }
}

export class NoiseReducedRecorder {
  private processor: NoiseReducer;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  
  constructor(processor: NoiseReducer) {
    this.processor = processor;
  }
  
  async start(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const processedStream = await this.processor.processMediaStream(stream);
    
    this.mediaRecorder = new MediaRecorder(processedStream);
    this.chunks = [];
    
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };
    
    this.mediaRecorder.start();
  }
  
  async stop(): Promise<Blob> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(new Blob());
        return;
      }
      
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        resolve(blob);
      };
      
      this.mediaRecorder.stop();
    });
  }
  
  pause(): void {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.pause();
    }
  }
  
  resume(): void {
    if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
      this.mediaRecorder.resume();
    }
  }
}

// ============================================================================
// Worker Support
// ============================================================================

export class NoiseReducerWorker {
  private worker: Worker | null = null;
  private messageId = 0;
  private pendingMessages: Map<number, {
    resolve: (value: any) => void;
    reject: (error: any) => void;
  }> = new Map();
  
  constructor(options: ProcessorOptions) {
    this.worker = new Worker('/noise-reducer-worker.js');
    
    this.worker.onmessage = (event) => {
      const { id, type, data, error } = event.data;
      
      const pending = this.pendingMessages.get(id);
      if (pending) {
        if (error) {
          pending.reject(error);
        } else {
          pending.resolve(data);
        }
        this.pendingMessages.delete(id);
      }
    };
    
    // Initialize worker
    this.postMessage({ type: 'init', options });
  }
  
  async process(buffer: Float32Array): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const id = this.messageId++;
      this.pendingMessages.set(id, { resolve, reject });
      
      this.worker!.postMessage(
        { id, type: 'process', buffer },
        [buffer.buffer]
      );
    });
  }
  
  postMessage(message: any): void {
    this.worker!.postMessage(message);
  }
  
  onMessage(handler: (data: any) => void): void {
    const originalHandler = this.worker!.onmessage;
    
    this.worker!.onmessage = (event) => {
      handler(event.data);
      if (originalHandler && this.worker) {
        originalHandler.call(this.worker, event);
      }
    };
  }
  
  processTransferable(
    buffer: ArrayBuffer, 
    callback: (result: ArrayBuffer) => void
  ): void {
    const id = this.messageId++;
    
    this.pendingMessages.set(id, {
      resolve: (data) => callback(data),
      reject: (error) => console.error('Worker error:', error)
    });
    
    this.worker!.postMessage(
      { id, type: 'process-transferable', buffer },
      [buffer]
    );
  }
  
  terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    
    this.pendingMessages.clear();
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

export function getPresetConfig(preset: NoiseReductionPreset): Partial<ProcessorOptions> {
  switch (preset) {
    case NoiseReductionPreset.VoiceChat:
      return {
        gateThreshold: -40,
        gateAttack: 5,
        gateRelease: 100,
        spectralEnabled: true,
        overSubtractionFactor: 2.5,
        spectralFloor: 0.1
      };
      
    case NoiseReductionPreset.MusicRecording:
      return {
        gateThreshold: -60,
        gateAttack: 0.1,
        gateRelease: 50,
        spectralEnabled: true,
        overSubtractionFactor: 1.5,
        spectralFloor: 0.05
      };
      
    case NoiseReductionPreset.PodcastRecording:
      return {
        gateThreshold: -45,
        gateAttack: 10,
        gateRelease: 200,
        spectralEnabled: true,
        overSubtractionFactor: 2.0,
        spectralFloor: 0.15
      };
      
    case NoiseReductionPreset.Aggressive:
      return {
        gateThreshold: -35,
        gateAttack: 1,
        gateRelease: 50,
        spectralEnabled: true,
        overSubtractionFactor: 3.0,
        spectralFloor: 0.2
      };
      
    case NoiseReductionPreset.Gentle:
      return {
        gateThreshold: -50,
        gateAttack: 20,
        gateRelease: 300,
        spectralEnabled: true,
        overSubtractionFactor: 1.2,
        spectralFloor: 0.05
      };
      
    default:
      return {};
  }
}

export async function processAudioFile(file: File): Promise<Blob> {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = await getAudioContext();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  const processor = new NoiseReducer({ 
    sampleRate: audioBuffer.sampleRate 
  });
  
  try {
    const processed = await processor.process(audioBuffer);
    
    // Convert back to blob
    const length = processed.length * processed.numberOfChannels;
    const buffer = new ArrayBuffer(length * 2); // 16-bit PCM
    const view = new DataView(buffer);
    
    let offset = 0;
    for (let i = 0; i < processed.length; i++) {
      for (let channel = 0; channel < processed.numberOfChannels; channel++) {
        const sample = processed.getChannelData(channel)[i];
        const int16 = Math.max(-32768, Math.min(32767, sample * 32768));
        view.setInt16(offset, int16, true);
        offset += 2;
      }
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
    
  } finally {
    processor.dispose();
  }
}

// ============================================================================
// Static Convenience Methods
// ============================================================================

export async function createMicrophoneProcessor(
  constraints?: MediaStreamConstraints
): Promise<MicrophoneProcessor> {
  return MicrophoneProcessor.create(constraints);
}