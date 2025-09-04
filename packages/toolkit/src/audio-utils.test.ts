/**
 * Tests for audio-utils.ts
 * 
 * These tests cover the browser-specific audio utilities and WASM integration helpers.
 * Note: Some Web Audio API features require a browser environment or polyfills.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  // Constants
  DEFAULT_SAMPLE_RATE,
  DEFAULT_FFT_SIZE,
  
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
  
  AudioUtils
} from './audio-utils';

// Import ProcessingErrorType enum values
enum ProcessingErrorType {
  INITIALIZATION_FAILED = 'INITIALIZATION_FAILED',
  PROCESSING_FAILED = 'PROCESSING_FAILED',
  BUFFER_UNDERRUN = 'BUFFER_UNDERRUN',
  BUFFER_OVERFLOW = 'BUFFER_OVERFLOW',
  INVALID_STATE = 'INVALID_STATE'
}

// Mock Web Audio API if not available (for Node.js testing)
class MockAudioContext {
  sampleRate = 48000;
  state = 'running' as AudioContextState;
  
  createBuffer(channels: number, length: number, sampleRate: number) {
    return new MockAudioBuffer(channels, length, sampleRate);
  }
  
  createScriptProcessor(bufferSize: number, inputChannels: number, outputChannels: number) {
    return {} as ScriptProcessorNode;
  }
  
  createMediaStreamSource(stream: MediaStream) {
    return {} as MediaStreamAudioSourceNode;
  }
  
  async resume() {
    this.state = 'running';
  }
  
  async decodeAudioData(buffer: ArrayBuffer) {
    // Simple mock - just create a buffer with some data
    return this.createBuffer(2, 48000, 48000);
  }
}

class MockAudioBuffer {
  constructor(
    public numberOfChannels: number,
    public length: number,
    public sampleRate: number
  ) {}
  
  get duration() {
    return this.length / this.sampleRate;
  }
  
  private channelData: Float32Array[] = [];
  
  getChannelData(channel: number): Float32Array {
    if (!this.channelData[channel]) {
      this.channelData[channel] = new Float32Array(this.length);
      // Fill with test data
      for (let i = 0; i < this.length; i++) {
        this.channelData[channel][i] = Math.sin(2 * Math.PI * i / 100) * 0.5;
      }
    }
    return this.channelData[channel];
  }
  
  copyToChannel(source: Float32Array, channel: number) {
    this.channelData[channel] = new Float32Array(source);
  }
}

class MockOfflineAudioContext extends MockAudioContext {
  constructor(channels: number, length: number, sampleRate: number) {
    super();
    this.sampleRate = sampleRate;
  }
}

// Setup mocks
beforeEach(() => {
  // Mock window APIs
  global.window = {
    AudioContext: MockAudioContext,
    OfflineAudioContext: MockOfflineAudioContext,
    isSecureContext: true,
    location: {
      protocol: 'https:',
      hostname: 'localhost'
    }
  } as any;
  
  // Make OfflineAudioContext available globally
  global.OfflineAudioContext = MockOfflineAudioContext as any;
  
  // Mock navigator APIs
  global.navigator = {
    mediaDevices: {
      getUserMedia: vi.fn().mockResolvedValue(new MediaStream())
    }
  } as any;
  
  // Mock document for download functionality
  global.document = {
    createElement: vi.fn((tag: string) => {
      if (tag === 'a') {
        return {
          href: '',
          download: '',
          click: vi.fn()
        };
      }
      return {};
    }),
    body: {
      appendChild: vi.fn(),
      removeChild: vi.fn()
    }
  } as any;
  
  // Mock URL
  global.URL = {
    createObjectURL: vi.fn(() => 'blob:mock-url'),
    revokeObjectURL: vi.fn()
  } as any;
  
  // Mock fetch
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    statusText: 'OK',
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(1024))
  });
  
  // Mock Worker
  global.Worker = class {
    constructor(public url: string, public options?: any) {}
    postMessage = vi.fn();
    terminate = vi.fn();
  } as any;
  
  // Mock SharedArrayBuffer
  global.SharedArrayBuffer = ArrayBuffer as any;
  
  // Mock MediaRecorder
  global.MediaRecorder = class {} as any;
  
  // Mock AudioWorkletNode
  global.AudioWorkletNode = class {
    constructor(context: any, name: string, options?: any) {}
  } as any;
  
  // Mock Blob with arrayBuffer method
  global.Blob = class MockBlob extends Blob {
    async arrayBuffer() {
      return new ArrayBuffer(1024);
    }
  } as any;
  
  // Also add ProcessingErrorType to global for the actual code
  (global as any).ProcessingErrorType = ProcessingErrorType;
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('Constants', () => {
  it('should export correct default values', () => {
    expect(DEFAULT_SAMPLE_RATE).toBe(48000);
    expect(DEFAULT_FFT_SIZE).toBe(512);
  });
});

describe('Audio Context Management', () => {
  it('should create and return audio context', () => {
    const ctx = getAudioContext();
    expect(ctx).toBeInstanceOf(MockAudioContext);
    expect(ctx.sampleRate).toBe(48000);
  });
  
  it('should reuse existing context', () => {
    const ctx1 = getAudioContext();
    const ctx2 = getAudioContext();
    expect(ctx1).toBe(ctx2);
  });
  
  it('should resume suspended context', async () => {
    const ctx = getAudioContext();
    ctx.state = 'suspended';
    await resumeAudioContext();
    expect(ctx.state).toBe('running');
  });
  
  it('should create offline context', () => {
    const ctx = createOfflineContext(2, 48000, 48000);
    expect(ctx).toBeInstanceOf(MockOfflineAudioContext);
    expect(ctx.sampleRate).toBe(48000);
  });
});

describe('Audio Buffer Operations', () => {
  it('should create audio buffer from Float32Array', () => {
    const data = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const buffer = createAudioBuffer(data);
    
    expect(buffer).toBeInstanceOf(MockAudioBuffer);
    expect(buffer.numberOfChannels).toBe(1);
    expect(buffer.length).toBe(4);
    expect(buffer.sampleRate).toBe(DEFAULT_SAMPLE_RATE);
  });
  
  it('should create multi-channel audio buffer', () => {
    const data = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    const buffer = createAudioBuffer(data, 48000, 2);
    
    expect(buffer.numberOfChannels).toBe(2);
    expect(buffer.length).toBe(4);
  });
  
  it('should extract channel data', () => {
    const buffer = new MockAudioBuffer(2, 100, 48000);
    const channelData = extractChannelData(buffer, 0);
    
    expect(channelData).toBeInstanceOf(Float32Array);
    expect(channelData.length).toBe(100);
  });
  
  it('should extract interleaved data', () => {
    const buffer = new MockAudioBuffer(2, 100, 48000);
    const interleaved = extractInterleavedData(buffer);
    
    expect(interleaved).toBeInstanceOf(Float32Array);
    expect(interleaved.length).toBe(200); // 2 channels * 100 samples
  });
  
  it('should clone audio buffer', () => {
    const source = new MockAudioBuffer(2, 100, 48000);
    const clone = cloneAudioBuffer(source);
    
    expect(clone).not.toBe(source);
    expect(clone.numberOfChannels).toBe(source.numberOfChannels);
    expect(clone.length).toBe(source.length);
    expect(clone.sampleRate).toBe(source.sampleRate);
  });
});

describe('Media Stream Handling', () => {
  it('should get user audio stream with default constraints', async () => {
    const stream = await getUserAudioStream();
    
    expect(stream).toBeInstanceOf(MediaStream);
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: DEFAULT_SAMPLE_RATE
      },
      video: false
    });
  });
  
  it('should create stream source', () => {
    const stream = new MediaStream();
    const source = createStreamSource(stream);
    
    expect(source).toBeDefined();
  });
  
  it('should create processor node', () => {
    const processor = createProcessor();
    
    expect(processor).toBeDefined();
  });
  
  it('should stop media stream tracks', () => {
    const track1 = { stop: vi.fn() };
    const track2 = { stop: vi.fn() };
    const stream = {
      getTracks: () => [track1, track2]
    } as any;
    
    stopStream(stream);
    
    expect(track1.stop).toHaveBeenCalled();
    expect(track2.stop).toHaveBeenCalled();
  });
});

describe('AudioWorklet Support', () => {
  it('should detect AudioWorklet support', () => {
    expect(isAudioWorkletSupported()).toBe(true);
  });
  
  it('should load AudioWorklet module', async () => {
    const ctx = getAudioContext();
    ctx.audioWorklet = {
      addModule: vi.fn().mockResolvedValue(undefined)
    };
    
    await loadAudioWorklet('processor.js');
    
    expect(ctx.audioWorklet.addModule).toHaveBeenCalledWith('processor.js');
  });
  
  it('should create AudioWorklet node', () => {
    const node = createAudioWorkletNode('test-processor');
    
    expect(node).toBeInstanceOf(AudioWorkletNode);
  });
});

describe('File I/O', () => {
  it('should load audio file', async () => {
    const file = new Blob(['audio data'], { type: 'audio/wav' });
    const buffer = await loadAudioFile(file);
    
    expect(buffer).toBeInstanceOf(MockAudioBuffer);
  });
  
  it('should load audio from URL', async () => {
    const buffer = await loadAudioFromURL('https://example.com/audio.wav');
    
    expect(fetch).toHaveBeenCalledWith('https://example.com/audio.wav');
    expect(buffer).toBeInstanceOf(MockAudioBuffer);
  });
  
  it('should convert audio buffer to WAV', () => {
    const buffer = new MockAudioBuffer(1, 100, 48000);
    const blob = audioBufferToWav(buffer);
    
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('audio/wav');
  });
  
  it('should download audio buffer', () => {
    const buffer = new MockAudioBuffer(1, 100, 48000);
    const createElementSpy = vi.spyOn(document, 'createElement');
    
    downloadAudioBuffer(buffer, 'test.wav');
    
    expect(createElementSpy).toHaveBeenCalledWith('a');
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });
});

describe('WASM Integration', () => {
  it('should prepare data for WASM', () => {
    const data = new Float32Array([0.5, NaN, Infinity, -Infinity, 1.5, -1.5]);
    const prepared = prepareForWasm(data);
    
    expect(prepared[0]).toBe(0.5);
    expect(prepared[1]).toBe(0);      // NaN -> 0
    expect(prepared[2]).toBe(0);      // Infinity -> 0
    expect(prepared[3]).toBe(0);      // -Infinity -> 0
    expect(prepared[4]).toBe(1);      // 1.5 -> clamped to 1
    expect(prepared[5]).toBe(-1);     // -1.5 -> clamped to -1
  });
  
  it('should create shared buffer when available', () => {
    const buffer = createSharedBuffer(1024);
    
    expect(buffer.byteLength).toBe(1024 * Float32Array.BYTES_PER_ELEMENT);
  });
  
  it('should fall back to regular ArrayBuffer when SharedArrayBuffer unavailable', () => {
    delete (global as any).SharedArrayBuffer;
    
    const buffer = createSharedBuffer(1024);
    
    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBe(1024 * Float32Array.BYTES_PER_ELEMENT);
  });
});

describe('Error Handling', () => {
  it('should create processing error', () => {
    const error = createProcessingError(
      ProcessingErrorType.PROCESSING_FAILED,
      'Test error',
      { detail: 'value' }
    );
    
    expect(error.type).toBe(ProcessingErrorType.PROCESSING_FAILED);
    expect(error.message).toBe('Test error');
    expect(error.details).toEqual({ detail: 'value' });
    expect(error.timestamp).toBeGreaterThan(0);
    expect(error.recoverable).toBe(true);
    expect(error.stack).toBeDefined();
  });
  
  it('should mark initialization errors as non-recoverable', () => {
    const error = createProcessingError(
      ProcessingErrorType.INITIALIZATION_FAILED,
      'Init error'
    );
    
    expect(error.recoverable).toBe(false);
  });
});

describe('Browser Capability Detection', () => {
  it('should detect browser capabilities', () => {
    const capabilities = detectCapabilities();
    
    expect(capabilities.audioContext).toBe(true);
    expect(capabilities.getUserMedia).toBe(true);
    expect(capabilities.audioWorklet).toBe(true);
    expect(capabilities.sharedArrayBuffer).toBe(true);
    expect(capabilities.offlineContext).toBe(true);
    expect(capabilities.mediaRecorder).toBe(true);
  });
  
  it('should check secure context', () => {
    expect(isSecureContext()).toBe(true);
    
    // Test non-secure context
    global.window.isSecureContext = false;
    global.location = {
      protocol: 'http:',
      hostname: 'example.com'
    } as any;
    expect(isSecureContext()).toBe(false);
    
    // Test localhost exception
    global.location.hostname = 'localhost';
    expect(isSecureContext()).toBe(true);
  });
});

describe('Web Worker Support', () => {
  it('should detect worker support', () => {
    expect(isWorkerSupported()).toBe(true);
  });
  
  it('should create processing worker', () => {
    const worker = createProcessingWorker('worker.js');
    
    expect(worker).toBeInstanceOf(Worker);
    expect((worker as any).url).toBe('worker.js');
  });
  
  it('should transfer data to worker with ownership transfer', () => {
    const worker = new Worker('test.js');
    const data = new Float32Array([1, 2, 3]);
    
    transferToWorker(worker, data, true);
    
    expect(worker.postMessage).toHaveBeenCalledWith(
      { audio: data },
      [data.buffer]
    );
  });
  
  it('should copy data to worker without transfer', () => {
    const worker = new Worker('test.js');
    const data = new Float32Array([1, 2, 3]);
    
    transferToWorker(worker, data, false);
    
    expect(worker.postMessage).toHaveBeenCalledWith({ audio: data });
  });
});

describe('Utility Functions', () => {
  it('should format time correctly', () => {
    expect(formatTime(0)).toBe('0:00');
    expect(formatTime(59)).toBe('0:59');
    expect(formatTime(60)).toBe('1:00');
    expect(formatTime(125)).toBe('2:05');
    expect(formatTime(3661)).toBe('61:01');
  });
  
  it('should convert buffer to string', () => {
    const buffer = new MockAudioBuffer(2, 48000, 48000);
    const str = bufferToString(buffer);
    
    expect(str).toBe('AudioBuffer[channels=2, length=48000, rate=48000Hz, duration=1.00s]');
  });
  
  it('should estimate latency', () => {
    const latency = estimateLatency(512, 48000);
    expect(latency).toBeCloseTo(10.667, 2); // 512/48000 * 1000
    
    const latency2 = estimateLatency(256, 44100);
    expect(latency2).toBeCloseTo(5.805, 2); // 256/44100 * 1000
  });
});

describe('AudioUtils Namespace', () => {
  it('should export all functions in namespace', () => {
    expect(AudioUtils.getAudioContext).toBe(getAudioContext);
    expect(AudioUtils.createAudioBuffer).toBe(createAudioBuffer);
    expect(AudioUtils.formatTime).toBe(formatTime);
    expect(AudioUtils.prepareForWasm).toBe(prepareForWasm);
    // ... and so on
  });
  
  it('should be the default export', () => {
    expect(AudioUtils).toBeDefined();
    expect(typeof AudioUtils.getAudioContext).toBe('function');
  });
});

describe('Edge Cases and Error Handling', () => {
  it('should handle invalid channel index gracefully', () => {
    const buffer = new MockAudioBuffer(2, 100, 48000);
    const data = extractChannelData(buffer, 10); // Channel 10 doesn't exist
    
    expect(data).toBeInstanceOf(Float32Array);
    // Should return last channel's data
  });
  
  it('should handle empty audio data', () => {
    const data = new Float32Array(0);
    const buffer = createAudioBuffer(data);
    
    expect(buffer.length).toBe(0);
  });
  
  it('should handle network errors when loading audio', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Not Found'
    });
    
    await expect(loadAudioFromURL('invalid.wav')).rejects.toThrow(
      'Failed to load audio from invalid.wav: Not Found'
    );
  });
  
  it('should handle AudioWorklet not supported', async () => {
    const ctx = getAudioContext();
    delete ctx.audioWorklet;
    
    await expect(loadAudioWorklet('test.js')).rejects.toThrow(
      'AudioWorklet not supported in this browser'
    );
  });
  
  it('should handle Worker not supported', () => {
    delete (global as any).Worker;
    
    expect(isWorkerSupported()).toBe(false);
    expect(createProcessingWorker('test.js')).toBeNull();
  });
});

describe('Integration Tests', () => {
  it('should handle complete audio processing pipeline', async () => {
    // 1. Get user media
    const stream = await getUserAudioStream();
    
    // 2. Create source
    const source = createStreamSource(stream);
    
    // 3. Create processor
    const processor = createProcessor(1, 1, 512);
    
    // 4. Process would happen here...
    
    // 5. Stop stream
    stopStream(stream);
    
    expect(source).toBeDefined();
    expect(processor).toBeDefined();
  });
  
  it('should handle file loading and conversion pipeline', async () => {
    // 1. Load file
    const file = new Blob(['audio'], { type: 'audio/wav' });
    const buffer = await loadAudioFile(file);
    
    // 2. Extract data for processing
    const channelData = extractChannelData(buffer);
    
    // 3. Prepare for WASM
    const wasmData = prepareForWasm(channelData);
    
    // 4. Could process with WASM here...
    
    // 5. Convert back to WAV
    const wavBlob = audioBufferToWav(buffer);
    
    expect(wasmData).toBeInstanceOf(Float32Array);
    expect(wavBlob).toBeInstanceOf(Blob);
  });
});