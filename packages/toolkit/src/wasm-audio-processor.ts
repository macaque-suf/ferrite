/**
 * WASM Audio Processor - Integration between audio-utils and WASM components
 * 
 * This module demonstrates how audio-utils.ts utilities work together with
 * the WASM noise reduction components to provide a complete audio processing pipeline.
 */

import init, { 
  NoiseReducer as WasmNoiseReducer, 
  greet 
} from '../../../dist/wasm/wasm_audio_ferrite';

import {
  // Audio context management
  getAudioContext,
  resumeAudioContext,
  createOfflineContext,
  
  // Buffer operations
  createAudioBuffer,
  extractChannelData,
  extractInterleavedData,
  cloneAudioBuffer,
  
  // File I/O
  loadAudioFile,
  loadAudioFromURL,
  audioBufferToWav,
  downloadAudioBuffer,
  
  // WASM preparation
  prepareForWasm,
  createSharedBuffer,
  
  // Error handling
  createProcessingError,
  
  // Utilities
  bufferToString,
  estimateLatency,
  
  // Media streaming
  getUserAudioStream,
  createStreamSource,
  createProcessor
} from './audio-utils';

// Import types
import type { 
  ProcessorConfig, 
  AudioFormat,
  ProcessingError,
  ProcessingErrorType
} from './types';

/**
 * Enhanced WASM Audio Processor that uses audio-utils for all browser operations
 */
export class WasmAudioProcessor {
  private wasmModule: WasmNoiseReducer | null = null;
  private initialized = false;
  private audioContext: AudioContext | null = null;
  private config: ProcessorConfig;
  
  constructor(config: Partial<ProcessorConfig> = {}) {
    this.config = {
      sampleRate: 48000,
      channels: 1,
      bitDepth: 16,
      ...config
    } as ProcessorConfig;
  }
  
  /**
   * Initialize WASM module and audio context
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      // 1. Initialize WASM module
      await init();
      
      // 2. Get or create audio context using audio-utils
      this.audioContext = getAudioContext();
      
      // 3. Resume context if needed (handles Chrome autoplay policy)
      await resumeAudioContext();
      
      // 4. Create WASM noise reducer
      this.wasmModule = new WasmNoiseReducer(this.config.sample_rate!);
      
      this.initialized = true;
      
      // Test WASM is working
      console.log(greet('WASM Audio Processor'));
    } catch (error) {
      throw createProcessingError(
        'INITIALIZATION_FAILED' as ProcessingErrorType,
        'Failed to initialize WASM audio processor',
        { error }
      );
    }
  }
  
  /**
   * Process audio file using WASM
   */
  async processAudioFile(file: File | Blob): Promise<AudioBuffer> {
    await this.initialize();
    
    // 1. Load audio file using audio-utils
    const inputBuffer = await loadAudioFile(file);
    
    console.log('Processing:', bufferToString(inputBuffer));
    console.log('Estimated latency:', estimateLatency(512, inputBuffer.sampleRate), 'ms');
    
    // 2. Process the buffer
    return this.processAudioBuffer(inputBuffer);
  }
  
  /**
   * Process audio from URL
   */
  async processAudioFromURL(url: string): Promise<AudioBuffer> {
    await this.initialize();
    
    // Load audio from URL using audio-utils
    const inputBuffer = await loadAudioFromURL(url);
    return this.processAudioBuffer(inputBuffer);
  }
  
  /**
   * Process AudioBuffer with WASM noise reduction
   */
  async processAudioBuffer(inputBuffer: AudioBuffer): Promise<AudioBuffer> {
    await this.initialize();
    
    if (!this.wasmModule) {
      throw createProcessingError(
        'INVALID_STATE' as ProcessingErrorType,
        'WASM module not initialized'
      );
    }
    
    try {
      // 1. Extract channel data using audio-utils
      const channelData = extractChannelData(inputBuffer, 0);
      
      // 2. Prepare data for WASM (ensure valid values, no NaN/Infinity)
      const preparedData = prepareForWasm(channelData);
      
      // 3. Process with WASM noise reducer
      const processedData = this.wasmModule.process(preparedData);
      
      // 4. Create output buffer using audio-utils
      const outputBuffer = createAudioBuffer(
        processedData,
        inputBuffer.sampleRate,
        1 // Mono for now
      );
      
      return outputBuffer;
    } catch (error) {
      throw createProcessingError(
        'PROCESSING_FAILED' as ProcessingErrorType,
        'Failed to process audio buffer',
        { error }
      );
    }
  }
  
  /**
   * Process multi-channel audio
   */
  async processMultiChannelBuffer(inputBuffer: AudioBuffer): Promise<AudioBuffer> {
    await this.initialize();
    
    // For stereo/multi-channel, process each channel separately
    const processedChannels: Float32Array[] = [];
    
    for (let channel = 0; channel < inputBuffer.numberOfChannels; channel++) {
      // Extract each channel
      const channelData = extractChannelData(inputBuffer, channel);
      
      // Prepare for WASM
      const preparedData = prepareForWasm(channelData);
      
      // Process with WASM
      const processed = this.wasmModule!.process(preparedData);
      processedChannels.push(new Float32Array(processed));
    }
    
    // Interleave channels if needed
    const outputLength = processedChannels[0].length;
    const interleavedData = new Float32Array(outputLength * inputBuffer.numberOfChannels);
    
    for (let i = 0; i < outputLength; i++) {
      for (let ch = 0; ch < inputBuffer.numberOfChannels; ch++) {
        interleavedData[i * inputBuffer.numberOfChannels + ch] = processedChannels[ch][i];
      }
    }
    
    // Create multi-channel output buffer
    return createAudioBuffer(
      interleavedData,
      inputBuffer.sampleRate,
      inputBuffer.numberOfChannels
    );
  }
  
  /**
   * Real-time processing with ScriptProcessor (legacy but widely supported)
   */
  async createRealtimeProcessor(stream: MediaStream): Promise<{
    source: MediaStreamAudioSourceNode;
    processor: ScriptProcessorNode;
    start: () => void;
    stop: () => void;
  }> {
    await this.initialize();
    
    const ctx = this.audioContext!;
    
    // Create nodes using audio-utils
    const source = createStreamSource(stream, ctx);
    const processor = createProcessor(1, 1, 512, ctx);
    
    // Process audio in real-time
    processor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);
      const outputData = event.outputBuffer.getChannelData(0);
      
      // Prepare for WASM
      const preparedData = prepareForWasm(inputData);
      
      // Process with WASM
      const processed = this.wasmModule!.process(preparedData);
      
      // Copy to output
      outputData.set(new Float32Array(processed));
    };
    
    return {
      source,
      processor,
      start: () => {
        source.connect(processor);
        processor.connect(ctx.destination);
      },
      stop: () => {
        source.disconnect();
        processor.disconnect();
      }
    };
  }
  
  /**
   * Offline processing for batch operations
   */
  async processOffline(
    inputBuffer: AudioBuffer,
    processingCallback?: (progress: number) => void
  ): Promise<AudioBuffer> {
    await this.initialize();
    
    // Create offline context using audio-utils
    const offlineCtx = createOfflineContext(
      inputBuffer.numberOfChannels,
      inputBuffer.length,
      inputBuffer.sampleRate
    );
    
    // Create source
    const source = offlineCtx.createBufferSource();
    source.buffer = inputBuffer;
    
    // Create script processor for offline processing
    const processor = offlineCtx.createScriptProcessor(512, 1, 1);
    let samplesProcessed = 0;
    
    processor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);
      const outputData = event.outputBuffer.getChannelData(0);
      
      // Process with WASM
      const preparedData = prepareForWasm(inputData);
      const processed = this.wasmModule!.process(preparedData);
      outputData.set(new Float32Array(processed));
      
      // Report progress
      samplesProcessed += inputData.length;
      if (processingCallback) {
        const progress = (samplesProcessed / inputBuffer.length) * 100;
        processingCallback(Math.min(progress, 100));
      }
    };
    
    // Connect nodes
    source.connect(processor);
    processor.connect(offlineCtx.destination);
    
    // Start processing
    source.start();
    
    // Render offline
    const renderedBuffer = await offlineCtx.startRendering();
    
    return renderedBuffer;
  }
  
  /**
   * Export processed audio as WAV file
   */
  async processAndDownload(
    inputBuffer: AudioBuffer,
    filename: string = 'processed-audio.wav'
  ): Promise<void> {
    // Process the audio
    const processedBuffer = await this.processAudioBuffer(inputBuffer);
    
    // Download using audio-utils
    downloadAudioBuffer(processedBuffer, filename);
  }
  
  /**
   * Get processing statistics
   */
  getProcessingInfo(): {
    initialized: boolean;
    sampleRate: number;
    latency: number;
    wasmReady: boolean;
  } {
    return {
      initialized: this.initialized,
      sampleRate: this.config.sample_rate!,
      latency: estimateLatency(512, this.config.sample_rate!),
      wasmReady: this.wasmModule !== null
    };
  }
  
  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.wasmModule) {
      // WASM module cleanup if needed
      this.wasmModule = null;
    }
    this.initialized = false;
  }
}

/**
 * Factory function for quick processing
 */
export async function processAudioWithWasm(
  input: File | Blob | AudioBuffer | string,
  options: Partial<ProcessorConfig> = {}
): Promise<AudioBuffer> {
  const processor = new WasmAudioProcessor(options);
  
  if (input instanceof AudioBuffer) {
    return processor.processAudioBuffer(input);
  } else if (typeof input === 'string') {
    return processor.processAudioFromURL(input);
  } else {
    return processor.processAudioFile(input);
  }
}

/**
 * Real-time processing helper
 */
export async function startRealtimeNoiseReduction(
  options: Partial<ProcessorConfig> = {}
): Promise<{
  processor: WasmAudioProcessor;
  stop: () => void;
}> {
  const processor = new WasmAudioProcessor(options);
  
  // Get user audio stream using audio-utils
  const stream = await getUserAudioStream({
    audio: {
      echoCancellation: false,
      noiseSuppression: false, // We'll do our own!
      autoGainControl: false,
      sampleRate: options.sample_rate || 48000
    }
  });
  
  // Create real-time processor
  const realtimeSetup = await processor.createRealtimeProcessor(stream);
  
  // Start processing
  realtimeSetup.start();
  
  return {
    processor,
    stop: () => {
      realtimeSetup.stop();
      stream.getTracks().forEach(track => track.stop());
    }
  };
}

// Export everything for convenience
export { 
  WasmAudioProcessor as default,
  // Re-export useful utilities from audio-utils
  prepareForWasm,
  createAudioBuffer,
  extractChannelData,
  loadAudioFile,
  loadAudioFromURL,
  downloadAudioBuffer
};