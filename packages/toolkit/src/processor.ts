import init, { NoiseReducer as WasmNoiseReducer } from './wasm/audio_processor_core';

export interface ProcessorOptions {
  sampleRate?: number;
  frameSize?: number;
  workerMode?: boolean;
}

export class NoiseReducer {
  private wasmModule: WasmNoiseReducer | null = null;
  private initialized = false;
  private sampleRate: number;

  constructor(options: ProcessorOptions = {}) {
    this.sampleRate = options.sampleRate || 48000;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    await init();
    this.wasmModule = new WasmNoiseReducer(this.sampleRate);
    this.initialized = true;
  }

  async process(audioBuffer: AudioBuffer): Promise<AudioBuffer> {
    if (!this.initialized) {
      await this.initialize();
    }

    const inputData = audioBuffer.getChannelData(0);
    const processed = this.wasmModule!.process(inputData);
    
    // Create output buffer
    const ctx = new AudioContext({ sampleRate: this.sampleRate });
    const outputBuffer = ctx.createBuffer(
      1,
      processed.length,
      this.sampleRate
    );
    outputBuffer.copyToChannel(new Float32Array(processed), 0);
    
    return outputBuffer;
  }

  static async removeNoise(audioBuffer: AudioBuffer): Promise<AudioBuffer> {
    const processor = new NoiseReducer({ 
      sampleRate: audioBuffer.sampleRate 
    });
    return processor.process(audioBuffer);
  }
}
