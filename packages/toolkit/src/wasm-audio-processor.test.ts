/**
 * Tests for WASM Audio Processor Integration
 * 
 * These tests verify the integration between audio-utils and WASM components
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  WasmAudioProcessor,
  processAudioWithWasm,
  startRealtimeNoiseReduction
} from './wasm-audio-processor';

// Mock the WASM module
vi.mock('./wasm/audio_processor_core', () => ({
  default: vi.fn().mockResolvedValue(undefined), // init function
  NoiseReducer: class MockNoiseReducer {
    constructor(public sampleRate: number) {}
    
    process(input: Float32Array): Float32Array {
      // Simple mock: reduce amplitude by 10% to simulate noise reduction
      const output = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        output[i] = input[i] * 0.9;
      }
      return output;
    }
  },
  greet: vi.fn((name: string) => `Hello, ${name}!`)
}));

// Mock audio-utils functions
vi.mock('./audio-utils', () => ({
  getAudioContext: vi.fn(() => ({
    sampleRate: 48000,
    state: 'running',
    createBuffer: vi.fn((channels, length, sampleRate) => ({
      numberOfChannels: channels,
      length,
      sampleRate,
      duration: length / sampleRate,
      getChannelData: vi.fn(() => new Float32Array(length)),
      copyToChannel: vi.fn()
    }))
  })),
  
  resumeAudioContext: vi.fn().mockResolvedValue(undefined),
  
  createOfflineContext: vi.fn((channels, length, sampleRate) => ({
    createBufferSource: vi.fn(() => ({
      buffer: null,
      connect: vi.fn(),
      start: vi.fn()
    })),
    createScriptProcessor: vi.fn(() => ({
      connect: vi.fn(),
      disconnect: vi.fn(),
      onaudioprocess: null
    })),
    destination: {},
    startRendering: vi.fn().mockResolvedValue({
      numberOfChannels: channels,
      length,
      sampleRate,
      getChannelData: vi.fn(() => new Float32Array(length))
    })
  })),
  
  createAudioBuffer: vi.fn((data, sampleRate, channels) => ({
    numberOfChannels: channels,
    length: data.length / channels,
    sampleRate,
    duration: data.length / channels / sampleRate,
    getChannelData: vi.fn(() => data)
  })),
  
  extractChannelData: vi.fn((buffer) => {
    const data = new Float32Array(100);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin(2 * Math.PI * i / 10) * 0.5;
    }
    return data;
  }),
  
  prepareForWasm: vi.fn((data) => {
    // Clean NaN and Infinity, clamp to [-1, 1]
    const cleaned = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const sample = data[i];
      if (!isFinite(sample)) {
        cleaned[i] = 0;
      } else {
        cleaned[i] = Math.max(-1, Math.min(1, sample));
      }
    }
    return cleaned;
  }),
  
  loadAudioFile: vi.fn().mockResolvedValue({
    numberOfChannels: 1,
    length: 48000,
    sampleRate: 48000,
    duration: 1,
    getChannelData: vi.fn(() => new Float32Array(48000))
  }),
  
  loadAudioFromURL: vi.fn().mockResolvedValue({
    numberOfChannels: 1,
    length: 48000,
    sampleRate: 48000,
    duration: 1,
    getChannelData: vi.fn(() => new Float32Array(48000))
  }),
  
  downloadAudioBuffer: vi.fn(),
  
  bufferToString: vi.fn((buffer) => 
    `AudioBuffer[channels=${buffer.numberOfChannels}, length=${buffer.length}, rate=${buffer.sampleRate}Hz]`
  ),
  
  estimateLatency: vi.fn((bufferSize, sampleRate) => (bufferSize / sampleRate) * 1000),
  
  createProcessingError: vi.fn((type, message, details) => ({
    type,
    message,
    details,
    timestamp: Date.now(),
    recoverable: type !== 'INITIALIZATION_FAILED'
  })),
  
  getUserAudioStream: vi.fn().mockResolvedValue({
    getTracks: () => [{
      stop: vi.fn()
    }]
  }),
  
  createStreamSource: vi.fn(() => ({
    connect: vi.fn(),
    disconnect: vi.fn()
  })),
  
  createProcessor: vi.fn(() => ({
    connect: vi.fn(),
    disconnect: vi.fn(),
    onaudioprocess: null
  }))
}));

describe('WasmAudioProcessor', () => {
  let processor: WasmAudioProcessor;
  
  beforeEach(() => {
    processor = new WasmAudioProcessor({
      sampleRate: 48000,
      channels: 1
    });
    vi.clearAllMocks();
  });
  
  describe('Initialization', () => {
    it('should initialize WASM module and audio context', async () => {
      await processor.initialize();
      
      const info = processor.getProcessingInfo();
      expect(info.initialized).toBe(true);
      expect(info.wasmReady).toBe(true);
      expect(info.sampleRate).toBe(48000);
    });
    
    it('should only initialize once', async () => {
      await processor.initialize();
      await processor.initialize();
      
      // Check that init was only called once
      const { default: init } = await import('./wasm/audio_processor_core');
      expect(init).toHaveBeenCalledTimes(1);
    });
  });
  
  describe('Audio Processing', () => {
    it('should process audio file', async () => {
      const file = new Blob(['audio data'], { type: 'audio/wav' });
      const result = await processor.processAudioFile(file);
      
      expect(result).toBeDefined();
      expect(result.numberOfChannels).toBe(1);
    });
    
    it('should process audio from URL', async () => {
      const result = await processor.processAudioFromURL('https://example.com/audio.wav');
      
      expect(result).toBeDefined();
      expect(result.numberOfChannels).toBe(1);
    });
    
    it('should process AudioBuffer', async () => {
      const inputBuffer = {
        numberOfChannels: 1,
        length: 100,
        sampleRate: 48000,
        duration: 100/48000,
        getChannelData: vi.fn(() => new Float32Array(100))
      } as any;
      
      const result = await processor.processAudioBuffer(inputBuffer);
      
      expect(result).toBeDefined();
      expect(result.numberOfChannels).toBe(1);
    });
    
    it('should handle multi-channel audio', async () => {
      const inputBuffer = {
        numberOfChannels: 2,
        length: 100,
        sampleRate: 48000,
        duration: 100/48000,
        getChannelData: vi.fn(() => new Float32Array(100))
      } as any;
      
      const result = await processor.processMultiChannelBuffer(inputBuffer);
      
      expect(result).toBeDefined();
      expect(result.numberOfChannels).toBe(2);
    });
  });
  
  describe('Real-time Processing', () => {
    it('should create real-time processor', async () => {
      const stream = {
        getTracks: () => [{ stop: vi.fn() }]
      } as any;
      
      const rtProcessor = await processor.createRealtimeProcessor(stream);
      
      expect(rtProcessor.source).toBeDefined();
      expect(rtProcessor.processor).toBeDefined();
      expect(rtProcessor.start).toBeInstanceOf(Function);
      expect(rtProcessor.stop).toBeInstanceOf(Function);
      
      // Test start/stop
      rtProcessor.start();
      rtProcessor.stop();
    });
  });
  
  describe('Offline Processing', () => {
    it('should process offline with progress callback', async () => {
      const inputBuffer = {
        numberOfChannels: 1,
        length: 1000,
        sampleRate: 48000,
        duration: 1000/48000,
        getChannelData: vi.fn(() => new Float32Array(1000))
      } as any;
      
      const progressValues: number[] = [];
      const result = await processor.processOffline(inputBuffer, (progress) => {
        progressValues.push(progress);
      });
      
      expect(result).toBeDefined();
      expect(result.numberOfChannels).toBe(1);
    });
  });
  
  describe('Export and Download', () => {
    it('should process and download audio', async () => {
      const inputBuffer = {
        numberOfChannels: 1,
        length: 100,
        sampleRate: 48000,
        duration: 100/48000,
        getChannelData: vi.fn(() => new Float32Array(100))
      } as any;
      
      await processor.processAndDownload(inputBuffer, 'test.wav');
      
      const { downloadAudioBuffer } = await import('./audio-utils');
      expect(downloadAudioBuffer).toHaveBeenCalledWith(
        expect.any(Object),
        'test.wav'
      );
    });
  });
  
  describe('Error Handling', () => {
    it('should handle uninitialized state', async () => {
      const inputBuffer = {
        numberOfChannels: 1,
        length: 100,
        sampleRate: 48000,
        getChannelData: vi.fn(() => new Float32Array(100))
      } as any;
      
      // Force uninitialized state by not calling initialize
      const newProcessor = new WasmAudioProcessor();
      // This should auto-initialize
      const result = await newProcessor.processAudioBuffer(inputBuffer);
      expect(result).toBeDefined();
    });
  });
  
  describe('Cleanup', () => {
    it('should dispose resources properly', async () => {
      await processor.initialize();
      processor.dispose();
      
      const info = processor.getProcessingInfo();
      expect(info.initialized).toBe(false);
      expect(info.wasmReady).toBe(false);
    });
  });
});

describe('Helper Functions', () => {
  it('should process audio with factory function', async () => {
    const file = new Blob(['audio'], { type: 'audio/wav' });
    const result = await processAudioWithWasm(file);
    
    expect(result).toBeDefined();
    expect(result.numberOfChannels).toBe(1);
  });
  
  it('should start real-time noise reduction', async () => {
    const { processor, stop } = await startRealtimeNoiseReduction({
      sampleRate: 44100
    });
    
    expect(processor).toBeInstanceOf(WasmAudioProcessor);
    expect(stop).toBeInstanceOf(Function);
    
    // Clean up
    stop();
  });
});

describe('Integration with audio-utils', () => {
  it('should use prepareForWasm to clean data', async () => {
    const { prepareForWasm } = await import('./audio-utils');
    
    const inputBuffer = {
      numberOfChannels: 1,
      length: 10,
      sampleRate: 48000,
      duration: 10/48000,
      getChannelData: vi.fn(() => new Float32Array([
        0.5, NaN, Infinity, -Infinity, 1.5, -1.5, 0, 0.7, -0.3, 0.9
      ]))
    } as any;
    
    await processor.processAudioBuffer(inputBuffer);
    
    expect(prepareForWasm).toHaveBeenCalled();
  });
  
  it('should handle all audio-utils functions correctly', async () => {
    const { 
      getAudioContext, 
      resumeAudioContext,
      extractChannelData,
      createAudioBuffer
    } = await import('./audio-utils');
    
    await processor.initialize();
    
    expect(getAudioContext).toHaveBeenCalled();
    expect(resumeAudioContext).toHaveBeenCalled();
    
    const buffer = {
      numberOfChannels: 1,
      length: 100,
      sampleRate: 48000,
      getChannelData: vi.fn(() => new Float32Array(100))
    } as any;
    
    await processor.processAudioBuffer(buffer);
    
    expect(extractChannelData).toHaveBeenCalled();
    expect(createAudioBuffer).toHaveBeenCalled();
  });
});