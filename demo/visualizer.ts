/**
 * Real-time audio spectrum visualizer for the noise reduction demo
 * Provides visual feedback showing the noise reduction system in action
 */

import type { 
  SpectralData, 
  NoiseProfile,
  ProcessingStats 
} from '../packages/toolkit/src/types';

export enum GateState {
  Closed = 'closed',
  Opening = 'opening',
  Open = 'open',
  Closing = 'closing'
}

export interface SpectrumData {
  frequencies: Float32Array;
  magnitudes: Float32Array;
  sampleRate: number;
  timestamp: number;
}

export interface VisualizationMetrics {
  fps: number;
  frameTime: number;
  droppedFrames: number;
}

export interface ColorScheme {
  background: string;
  spectrum: string | CanvasGradient;
  noiseProfile: string;
  grid: string;
  text: string;
  gateOpen: string;
  gateClosed: string;
  peakHold: string;
}

export interface VisualizerConfig {
  canvasId: string;
  fftSize?: number;
  smoothingTimeConstant?: number;
  minDecibels?: number;
  maxDecibels?: number;
  colorScheme?: ColorScheme;
}

interface VisualizerState {
  isRunning: boolean;
  currentSpectrum: Float32Array;
  noiseProfile: Float32Array | null;
  peakHolds: Float32Array;
  peakHoldTimer: number[];
  gateState: GateState;
  gainReduction: number;
  frameCount: number;
  lastFrameTime: number;
  mouseX: number;
  mouseY: number;
  isFrozen: boolean;
}

export class SpectrumVisualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private analyser: AnalyserNode | null = null;
  // Removed audioContext - we'll receive an external AnalyserNode instead
  
  private config: Required<VisualizerConfig>;
  private state: VisualizerState;
  private animationId: number | null = null;
  
  private frequencyScale: 'linear' | 'log' = 'log';
  private showPeaks = true;
  private showGrid = true;
  private showLegend = true;
  
  private colorScheme: ColorScheme;
  private spectrumGradient: CanvasGradient | null = null;
  
  private readonly PEAK_HOLD_TIME = 1500; // ms
  private readonly PEAK_DECAY_RATE = 0.5; // dB/frame
  private readonly MIN_FREQ = 20; // Hz
  private readonly MAX_FREQ = 20000; // Hz
  
  constructor(config: VisualizerConfig) {
    const canvas = document.getElementById(config.canvasId);
    if (!canvas || !(canvas instanceof HTMLCanvasElement)) {
      throw new Error(`Canvas element with id "${config.canvasId}" not found`);
    }
    
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }
    this.ctx = ctx;
    
    // Set default configuration
    this.config = {
      canvasId: config.canvasId,
      fftSize: config.fftSize || 2048,
      smoothingTimeConstant: config.smoothingTimeConstant || 0.8,
      minDecibels: config.minDecibels || -90,
      maxDecibels: config.maxDecibels || -10,
      colorScheme: config.colorScheme || this.getDefaultColorScheme()
    };
    
    this.colorScheme = this.config.colorScheme;
    
    // Initialize state
    const binCount = this.config.fftSize / 2;
    this.state = {
      isRunning: false,
      currentSpectrum: new Float32Array(binCount),
      noiseProfile: null,
      peakHolds: new Float32Array(binCount),
      peakHoldTimer: new Array(binCount).fill(0),
      gateState: GateState.Closed,
      gainReduction: 0,
      frameCount: 0,
      lastFrameTime: performance.now(),
      mouseX: -1,
      mouseY: -1,
      isFrozen: false
    };
    
    this.setupCanvas();
    this.setupEventListeners();
  }
  
  private getDefaultColorScheme(): ColorScheme {
    return {
      background: '#1a1a1a',
      spectrum: '#00ff88',
      noiseProfile: 'rgba(255, 100, 100, 0.5)',
      grid: 'rgba(255, 255, 255, 0.1)',
      text: '#ffffff',
      gateOpen: '#00ff00',
      gateClosed: '#ff0000',
      peakHold: 'rgba(255, 255, 0, 0.7)'
    };
  }
  
  private setupCanvas(): void {
    this.resize();
    
    // Setup gradient for spectrum
    this.spectrumGradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
    this.spectrumGradient.addColorStop(0, '#ff0000');
    this.spectrumGradient.addColorStop(0.25, '#ffff00');
    this.spectrumGradient.addColorStop(0.5, '#00ff00');
    this.spectrumGradient.addColorStop(0.75, '#00ffff');
    this.spectrumGradient.addColorStop(1, '#0000ff');
  }
  
  private setupEventListeners(): void {
    window.addEventListener('resize', () => this.resize());
    
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.state.mouseX = e.clientX - rect.left;
      this.state.mouseY = e.clientY - rect.top;
    });
    
    this.canvas.addEventListener('mouseleave', () => {
      this.state.mouseX = -1;
      this.state.mouseY = -1;
    });
    
    this.canvas.addEventListener('click', () => {
      this.state.isFrozen = !this.state.isFrozen;
    });
  }
  
  async initialize(): Promise<void> {
    // Analyser will be set via connectAudioNode instead
    // This allows the parent to control the AudioContext
  }
  
  connectAudioNode(analyser: AnalyserNode): void {
    // Accept an AnalyserNode directly instead of connecting to it
    this.analyser = analyser;
    // Apply our configuration to the provided analyser
    this.analyser.fftSize = this.config.fftSize;
    this.analyser.smoothingTimeConstant = this.config.smoothingTimeConstant;
    this.analyser.minDecibels = this.config.minDecibels;
    this.analyser.maxDecibels = this.config.maxDecibels;
  }
  
  start(): void {
    if (this.state.isRunning) return;
    
    this.state.isRunning = true;
    this.state.lastFrameTime = performance.now();
    this.animate();
  }
  
  stop(): void {
    this.state.isRunning = false;
    
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
  
  private animate = (): void => {
    if (!this.state.isRunning) return;
    
    const currentTime = performance.now();
    const deltaTime = currentTime - this.state.lastFrameTime;
    
    if (!this.state.isFrozen) {
      this.updateSpectrum();
      this.updatePeakHolds(deltaTime);
    }
    
    this.draw();
    
    this.state.frameCount++;
    this.state.lastFrameTime = currentTime;
    
    this.animationId = requestAnimationFrame(this.animate);
  };
  
  private updateSpectrum(): void {
    if (!this.analyser) return;
    
    const dataArray = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatFrequencyData(dataArray);
    this.state.currentSpectrum = dataArray;
  }
  
  private updatePeakHolds(deltaTime: number): void {
    const spectrum = this.state.currentSpectrum;
    const peaks = this.state.peakHolds;
    const timers = this.state.peakHoldTimer;
    
    for (let i = 0; i < spectrum.length; i++) {
      if (spectrum[i] > peaks[i]) {
        peaks[i] = spectrum[i];
        timers[i] = this.PEAK_HOLD_TIME;
      } else if (timers[i] > 0) {
        timers[i] -= deltaTime;
      } else {
        peaks[i] -= this.PEAK_DECAY_RATE;
        if (peaks[i] < spectrum[i]) {
          peaks[i] = spectrum[i];
        }
      }
    }
  }
  
  draw(): void {
    this.clear();
    
    if (this.showGrid) {
      this.drawGrid();
    }
    
    this.drawSpectrum(this.state.currentSpectrum);
    
    if (this.state.noiseProfile) {
      this.drawNoiseProfile(this.state.noiseProfile);
    }
    
    if (this.showPeaks) {
      this.drawPeakHolds();
    }
    
    this.drawAxis();
    this.drawLabels();
    
    if (this.showLegend) {
      this.drawLegend();
    }
    
    this.drawStatusIndicators();
    
    if (this.state.mouseX >= 0 && this.state.mouseY >= 0) {
      this.drawCursorInfo();
    }
  }
  
  private clear(): void {
    this.ctx.fillStyle = this.colorScheme.background;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }
  
  private drawGrid(): void {
    this.ctx.strokeStyle = this.colorScheme.grid;
    this.ctx.lineWidth = 0.5;
    
    // Horizontal lines (dB levels)
    const dbSteps = [-80, -60, -40, -20, 0];
    for (const db of dbSteps) {
      const y = this.amplitudeToY(db);
      this.ctx.beginPath();
      this.ctx.moveTo(50, y);
      this.ctx.lineTo(this.canvas.width - 20, y);
      this.ctx.stroke();
    }
    
    // Vertical lines (frequency markers)
    const freqSteps = this.frequencyScale === 'log' 
      ? [100, 1000, 10000]
      : [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000];
      
    for (const freq of freqSteps) {
      const x = this.frequencyToX(freq);
      this.ctx.beginPath();
      this.ctx.moveTo(x, 20);
      this.ctx.lineTo(x, this.canvas.height - 40);
      this.ctx.stroke();
    }
  }
  
  private drawAxis(): void {
    this.ctx.strokeStyle = this.colorScheme.text;
    this.ctx.lineWidth = 1;
    
    // Y-axis
    this.ctx.beginPath();
    this.ctx.moveTo(50, 20);
    this.ctx.lineTo(50, this.canvas.height - 40);
    this.ctx.stroke();
    
    // X-axis
    this.ctx.beginPath();
    this.ctx.moveTo(50, this.canvas.height - 40);
    this.ctx.lineTo(this.canvas.width - 20, this.canvas.height - 40);
    this.ctx.stroke();
  }
  
  private drawSpectrum(data: Float32Array): void {
    if (!this.audioContext) return;
    
    const gradient = this.spectrumGradient || this.colorScheme.spectrum;
    this.ctx.fillStyle = gradient;
    this.ctx.strokeStyle = gradient;
    this.ctx.lineWidth = 2;
    
    const nyquist = this.audioContext.sampleRate / 2;
    const binWidth = nyquist / data.length;
    
    this.ctx.beginPath();
    let started = false;
    
    for (let i = 0; i < data.length; i++) {
      const freq = i * binWidth;
      if (freq < this.MIN_FREQ || freq > this.MAX_FREQ) continue;
      
      const x = this.frequencyToX(freq);
      const y = this.amplitudeToY(data[i]);
      
      if (!started) {
        this.ctx.moveTo(x, y);
        started = true;
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    
    this.ctx.stroke();
    
    // Fill area under spectrum
    if (started) {
      this.ctx.lineTo(this.canvas.width - 20, this.canvas.height - 40);
      this.ctx.lineTo(50, this.canvas.height - 40);
      this.ctx.closePath();
      this.ctx.globalAlpha = 0.3;
      this.ctx.fill();
      this.ctx.globalAlpha = 1;
    }
  }
  
  private drawNoiseProfile(profile: Float32Array): void {
    if (!this.audioContext) return;
    
    this.ctx.strokeStyle = this.colorScheme.noiseProfile;
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([5, 5]);
    
    const nyquist = this.audioContext.sampleRate / 2;
    const binWidth = nyquist / profile.length;
    
    this.ctx.beginPath();
    let started = false;
    
    for (let i = 0; i < profile.length; i++) {
      const freq = i * binWidth;
      if (freq < this.MIN_FREQ || freq > this.MAX_FREQ) continue;
      
      const x = this.frequencyToX(freq);
      const y = this.amplitudeToY(profile[i]);
      
      if (!started) {
        this.ctx.moveTo(x, y);
        started = true;
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    
    this.ctx.stroke();
    this.ctx.setLineDash([]);
  }
  
  private drawPeakHolds(): void {
    if (!this.audioContext) return;
    
    this.ctx.strokeStyle = this.colorScheme.peakHold;
    this.ctx.lineWidth = 1;
    
    const nyquist = this.audioContext.sampleRate / 2;
    const binWidth = nyquist / this.state.peakHolds.length;
    
    for (let i = 0; i < this.state.peakHolds.length; i++) {
      if (this.state.peakHoldTimer[i] <= 0) continue;
      
      const freq = i * binWidth;
      if (freq < this.MIN_FREQ || freq > this.MAX_FREQ) continue;
      
      const x = this.frequencyToX(freq);
      const y = this.amplitudeToY(this.state.peakHolds[i]);
      
      this.ctx.beginPath();
      this.ctx.moveTo(x - 2, y);
      this.ctx.lineTo(x + 2, y);
      this.ctx.stroke();
    }
  }
  
  private drawLabels(): void {
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.font = '10px monospace';
    this.ctx.textAlign = 'right';
    
    // Y-axis labels (dB)
    const dbSteps = [-80, -60, -40, -20, 0];
    for (const db of dbSteps) {
      const y = this.amplitudeToY(db);
      this.ctx.fillText(`${db} dB`, 45, y + 3);
    }
    
    // X-axis labels (frequency)
    this.ctx.textAlign = 'center';
    const freqLabels = this.frequencyScale === 'log'
      ? [[100, '100'], [1000, '1k'], [10000, '10k']]
      : [[2000, '2k'], [4000, '4k'], [6000, '6k'], [8000, '8k'], 
         [10000, '10k'], [12000, '12k'], [14000, '14k'], [16000, '16k']];
    
    for (const [freq, label] of freqLabels) {
      const x = this.frequencyToX(freq as number);
      this.ctx.fillText(`${label}Hz`, x, this.canvas.height - 25);
    }
  }
  
  private drawLegend(): void {
    const legendX = this.canvas.width - 150;
    const legendY = 30;
    
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    this.ctx.fillRect(legendX - 10, legendY - 15, 140, 100);
    
    this.ctx.font = '11px monospace';
    let y = legendY;
    
    // Spectrum
    this.ctx.fillStyle = '#00ff88';
    this.ctx.fillRect(legendX, y, 15, 10);
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.textAlign = 'left';
    this.ctx.fillText('Spectrum', legendX + 20, y + 8);
    y += 20;
    
    // Noise Profile
    if (this.state.noiseProfile) {
      this.ctx.fillStyle = this.colorScheme.noiseProfile;
      this.ctx.fillRect(legendX, y, 15, 10);
      this.ctx.fillStyle = this.colorScheme.text;
      this.ctx.fillText('Noise Profile', legendX + 20, y + 8);
      y += 20;
    }
    
    // Peak Holds
    if (this.showPeaks) {
      this.ctx.fillStyle = this.colorScheme.peakHold;
      this.ctx.fillRect(legendX, y, 15, 10);
      this.ctx.fillStyle = this.colorScheme.text;
      this.ctx.fillText('Peak Hold', legendX + 20, y + 8);
      y += 20;
    }
    
    // Gate Status
    this.ctx.fillStyle = this.state.gateState === GateState.Open 
      ? this.colorScheme.gateOpen 
      : this.colorScheme.gateClosed;
    this.ctx.fillRect(legendX, y, 15, 10);
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.fillText(`Gate: ${this.state.gateState}`, legendX + 20, y + 8);
  }
  
  private drawStatusIndicators(): void {
    const statusX = 60;
    const statusY = 30;
    
    this.ctx.font = '12px monospace';
    this.ctx.textAlign = 'left';
    
    // Gate indicator
    const gateColor = this.state.gateState === GateState.Open 
      ? this.colorScheme.gateOpen 
      : this.state.gateState === GateState.Closed
      ? this.colorScheme.gateClosed
      : '#ffff00';
    
    this.ctx.fillStyle = gateColor;
    this.ctx.beginPath();
    this.ctx.arc(statusX, statusY, 6, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.fillText(`Gate: ${this.state.gateState}`, statusX + 15, statusY + 4);
    
    // Gain reduction
    if (this.state.gainReduction !== 0) {
      this.ctx.fillText(
        `Reduction: ${this.state.gainReduction.toFixed(1)} dB`, 
        statusX, 
        statusY + 20
      );
    }
    
    // Frozen indicator
    if (this.state.isFrozen) {
      this.ctx.fillStyle = '#ff0000';
      this.ctx.fillText('FROZEN', statusX, statusY + 40);
    }
    
    // FPS counter
    const fps = this.calculateFPS();
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.fillText(`FPS: ${fps}`, statusX, this.canvas.height - 50);
  }
  
  private drawCursorInfo(): void {
    if (!this.audioContext) return;
    
    const freq = this.xToFrequency(this.state.mouseX);
    const amp = this.yToAmplitude(this.state.mouseY);
    
    if (freq < this.MIN_FREQ || freq > this.MAX_FREQ) return;
    
    // Draw crosshair
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    this.ctx.lineWidth = 0.5;
    this.ctx.setLineDash([2, 2]);
    
    this.ctx.beginPath();
    this.ctx.moveTo(this.state.mouseX, 20);
    this.ctx.lineTo(this.state.mouseX, this.canvas.height - 40);
    this.ctx.stroke();
    
    this.ctx.beginPath();
    this.ctx.moveTo(50, this.state.mouseY);
    this.ctx.lineTo(this.canvas.width - 20, this.state.mouseY);
    this.ctx.stroke();
    
    this.ctx.setLineDash([]);
    
    // Draw info box
    const infoText = `${freq.toFixed(0)} Hz, ${amp.toFixed(1)} dB`;
    this.ctx.font = '11px monospace';
    const textWidth = this.ctx.measureText(infoText).width;
    
    const boxX = this.state.mouseX + 10;
    const boxY = this.state.mouseY - 25;
    
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    this.ctx.fillRect(boxX, boxY, textWidth + 10, 20);
    
    this.ctx.fillStyle = this.colorScheme.text;
    this.ctx.textAlign = 'left';
    this.ctx.fillText(infoText, boxX + 5, boxY + 14);
  }
  
  private calculateFPS(): number {
    if (this.state.frameCount < 10) return 0;
    
    const currentTime = performance.now();
    const avgFrameTime = (currentTime - this.state.lastFrameTime) / this.state.frameCount;
    return Math.round(1000 / avgFrameTime);
  }
  
  // Coordinate conversion methods
  private frequencyToX(freq: number): number {
    const leftMargin = 50;
    const rightMargin = 20;
    const width = this.canvas.width - leftMargin - rightMargin;
    
    if (this.frequencyScale === 'log') {
      const minLog = Math.log10(this.MIN_FREQ);
      const maxLog = Math.log10(this.MAX_FREQ);
      const freqLog = Math.log10(Math.max(freq, this.MIN_FREQ));
      const normalized = (freqLog - minLog) / (maxLog - minLog);
      return leftMargin + normalized * width;
    } else {
      const normalized = (freq - this.MIN_FREQ) / (this.MAX_FREQ - this.MIN_FREQ);
      return leftMargin + normalized * width;
    }
  }
  
  private amplitudeToY(amp: number): number {
    const topMargin = 20;
    const bottomMargin = 40;
    const height = this.canvas.height - topMargin - bottomMargin;
    
    const normalized = (amp - this.config.minDecibels) / 
                      (this.config.maxDecibels - this.config.minDecibels);
    return topMargin + (1 - normalized) * height;
  }
  
  private xToFrequency(x: number): number {
    const leftMargin = 50;
    const rightMargin = 20;
    const width = this.canvas.width - leftMargin - rightMargin;
    
    const normalized = (x - leftMargin) / width;
    
    if (this.frequencyScale === 'log') {
      const minLog = Math.log10(this.MIN_FREQ);
      const maxLog = Math.log10(this.MAX_FREQ);
      const freqLog = minLog + normalized * (maxLog - minLog);
      return Math.pow(10, freqLog);
    } else {
      return this.MIN_FREQ + normalized * (this.MAX_FREQ - this.MIN_FREQ);
    }
  }
  
  private yToAmplitude(y: number): number {
    const topMargin = 20;
    const bottomMargin = 40;
    const height = this.canvas.height - topMargin - bottomMargin;
    
    const normalized = 1 - (y - topMargin) / height;
    return this.config.minDecibels + 
           normalized * (this.config.maxDecibels - this.config.minDecibels);
  }
  
  // Public update methods
  updateNoiseProfile(profile: Float32Array): void {
    this.state.noiseProfile = profile;
  }
  
  updateGateState(state: GateState): void {
    this.state.gateState = state;
  }
  
  updateGainReduction(gainDb: number): void {
    this.state.gainReduction = gainDb;
  }
  
  // Configuration methods
  setColorScheme(scheme: ColorScheme): void {
    this.colorScheme = scheme;
    this.setupCanvas();
  }
  
  setFrequencyScale(scale: 'linear' | 'log'): void {
    this.frequencyScale = scale;
  }
  
  setShowPeaks(enabled: boolean): void {
    this.showPeaks = enabled;
  }
  
  setShowGrid(enabled: boolean): void {
    this.showGrid = enabled;
  }
  
  // Utility methods
  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    
    this.ctx.scale(dpr, dpr);
    
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    
    if (this.state.isRunning) {
      this.draw();
    }
  }
  
  screenshot(): string {
    return this.canvas.toDataURL('image/png');
  }
  
  destroy(): void {
    this.stop();
    window.removeEventListener('resize', () => this.resize());
    
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    
    this.audioContext = null;
  }
}
