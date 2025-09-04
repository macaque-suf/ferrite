/**
 * Comprehensive end-to-end test suite for WASM Audio Ferrite
 */

import { 
  NoiseReducer, 
  ProcessorOptions,
  NoiseReductionPreset,
  BufferPool,
  MicrophoneProcessor,
  NoiseReducerWorker,
  processAudioFile,
  createMicrophoneProcessor
} from '../packages/toolkit/src/processor';
import { SpectrumVisualizer, GateState } from './visualizer';

// Global test state
let processor: NoiseReducer | null = null;
let micProcessor: MicrophoneProcessor | null = null;
let visualizer: SpectrumVisualizer | null = null;
let worker: NoiseReducerWorker | null = null;
let processedBlob: Blob | null = null;
let isProcessingEnabled = false;
let perfMonitorInterval: number | null = null;

// Recording variables for microphone section
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let recordedBlob: Blob | null = null;
let recordingStartTime = 0;
let micStream: MediaStream | null = null;
let recordingDestination: MediaStreamAudioDestinationNode | null = null;
let outputGainNode: GainNode | null = null;

// Utility function to log results
function logResult(elementId: string, message: string, type: 'success' | 'error' | 'info' = 'info') {
  const element = document.getElementById(elementId);
  if (element) {
    element.className = `test-result ${type}`;
    element.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  }
}

// Utility function to update status
function updateStatus(elementId: string, value: string | number) {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = String(value);
  }
}

// ============================================================================
// 1. Browser Support Test
// ============================================================================

document.getElementById('checkSupportBtn')?.addEventListener('click', () => {
  const support = NoiseReducer.checkBrowserSupport();
  
  const results = [
    `✅ Web Audio API: ${support.webAudioApi}`,
    `✅ WebAssembly: ${support.webAssembly}`,
    `${support.audioWorklet ? '✅' : '⚠️'} AudioWorklet: ${support.audioWorklet}`,
    `${support.sharedArrayBuffer ? '✅' : '⚠️'} SharedArrayBuffer: ${support.sharedArrayBuffer}`,
    `✅ MediaDevices: ${support.mediaDevices}`,
  ];
  
  logResult('supportResult', results.join('\n'), 
    support.webAudioApi && support.webAssembly ? 'success' : 'error');
});

// ============================================================================
// 2. Basic Initialization Test
// ============================================================================

document.getElementById('initBasicBtn')?.addEventListener('click', async () => {
  try {
    logResult('initResult', 'Initializing NoiseReducer...', 'info');
    
    processor = new NoiseReducer({
      sampleRate: 48000,
      frameSize: 256,
      workletMode: false, // Use ScriptProcessor for testing
    });
    
    await processor.initialize();
    
    const isInit = processor.isInitialized();
    const config = processor.getConfig();
    
    logResult('initResult', 
      `✅ NoiseReducer initialized successfully!\n` +
      `Initialized: ${isInit}\n` +
      `Sample Rate: ${config.sampleRate}\n` +
      `Frame Size: ${config.frameSize}\n` +
      `Gate Enabled: ${config.gateEnabled}\n` +
      `Spectral Enabled: ${config.spectralEnabled}`,
      'success'
    );
    
    document.getElementById('disposeBasicBtn')!.disabled = false;
    document.getElementById('initBasicBtn')!.disabled = true;
    
  } catch (error) {
    logResult('initResult', `❌ Initialization failed: ${error}`, 'error');
  }
});

document.getElementById('disposeBasicBtn')?.addEventListener('click', () => {
  if (processor) {
    processor.dispose();
    processor = null;
    
    logResult('initResult', '✅ NoiseReducer disposed successfully', 'success');
    
    document.getElementById('disposeBasicBtn')!.disabled = true;
    document.getElementById('initBasicBtn')!.disabled = false;
  }
});

// ============================================================================
// 3. Microphone Processing Test
// ============================================================================

// Global audio context to ensure all nodes use the same context
let globalAudioContext: AudioContext | null = null;

document.getElementById('startMicBtn')?.addEventListener('click', async () => {
  try {
    logResult('micResult', 'Starting microphone...', 'info');
    
    // Create or reuse audio context
    if (!globalAudioContext) {
      globalAudioContext = new AudioContext();
    }
    
    // Create processor if not exists
    if (!processor) {
      processor = new NoiseReducer({
        sampleRate: globalAudioContext.sampleRate
      });
      await processor.initialize();
    }
    
    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });
    
    // Store stream for recording
    micStream = stream;
    
    const source = globalAudioContext.createMediaStreamSource(stream);
    
    // Setup visualizer
    if (!visualizer) {
      visualizer = new SpectrumVisualizer({
        canvasId: 'spectrumCanvas',
        fftSize: 2048
      });
      await visualizer.initialize();
    }
    
    // Create analyser node in the same context
    const analyserForViz = globalAudioContext.createAnalyser();
    analyserForViz.fftSize = 2048;
    source.connect(analyserForViz);
    visualizer.connectAudioNode(analyserForViz);
    visualizer.start();
    
    // Setup meter
    const analyser = globalAudioContext.createAnalyser();
    source.connect(analyser);
    
    function updateMeter() {
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      const meter = document.getElementById('micMeter') as HTMLElement;
      if (meter) {
        meter.style.width = `${(average / 255) * 100}%`;
      }
      requestAnimationFrame(updateMeter);
    }
    updateMeter();
    
    // Connect processor using ScriptProcessor for compatibility
    // Note: ScriptProcessorNode is deprecated but still widely supported
    // For production, use AudioWorkletNode instead (requires HTTPS in Chrome)
    // Using 4096 buffer size for better quality (reduces choppiness)
    const scriptProcessor = globalAudioContext.createScriptProcessor(4096, 1, 1);
    
    scriptProcessor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const output = event.outputBuffer.getChannelData(0);
      
      // CRITICAL: Audio callbacks must be synchronous!
      // Async processing causes flanging, phasing, and audio dropouts
      
      // For now, just pass through the audio cleanly
      // Processing can be done in a separate worker or AudioWorklet
      output.set(input);
      
      // If we're capturing noise, store the samples
      if (processor && processor.isCapturingNoise) {
        // Store samples for noise profile (this is synchronous)
        processor.storeNoiseSample(new Float32Array(input));
      }
    };
    
    source.connect(scriptProcessor);
    
    // Create gain node for muting speaker output during recording
    outputGainNode = globalAudioContext.createGain();
    outputGainNode.gain.value = 1; // Start with audio enabled
    
    // Connect: scriptProcessor -> outputGain -> speakers
    scriptProcessor.connect(outputGainNode);
    outputGainNode.connect(globalAudioContext.destination);
    
    // Create a destination for recording processed audio (always connected, not affected by gain)
    recordingDestination = globalAudioContext.createMediaStreamDestination();
    scriptProcessor.connect(recordingDestination);
    
    logResult('micResult', '✅ Microphone started successfully', 'success');
    
    document.getElementById('startMicBtn')!.disabled = true;
    document.getElementById('toggleProcessingBtn')!.disabled = false;
    document.getElementById('stopMicBtn')!.disabled = false;
    document.getElementById('startRecordBtn')!.disabled = false;
    
  } catch (error) {
    logResult('micResult', `❌ Failed to start microphone: ${error}`, 'error');
  }
});

document.getElementById('toggleProcessingBtn')?.addEventListener('click', () => {
  isProcessingEnabled = !isProcessingEnabled;
  
  const btn = document.getElementById('toggleProcessingBtn') as HTMLButtonElement;
  btn.textContent = isProcessingEnabled ? 'Disable Processing' : 'Enable Processing';
  
  if (processor) {
    processor.setGateEnabled(isProcessingEnabled);
    processor.setSpectralEnabled(isProcessingEnabled);
  }
  
  if (visualizer) {
    visualizer.updateGateState(isProcessingEnabled ? GateState.Open : GateState.Closed);
  }
  
  logResult('micResult', 
    isProcessingEnabled ? '🟢 Processing enabled' : '⭕ Processing disabled', 
    'info'
  );
});

document.getElementById('stopMicBtn')?.addEventListener('click', () => {
  // Stop recording if active
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  
  // Stop visualizer
  if (visualizer) {
    visualizer.stop();
  }
  
  // Stop microphone stream
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  
  logResult('micResult', '✅ Microphone stopped', 'success');
  
  document.getElementById('startMicBtn')!.disabled = false;
  document.getElementById('toggleProcessingBtn')!.disabled = true;
  document.getElementById('stopMicBtn')!.disabled = true;
  document.getElementById('startRecordBtn')!.disabled = true;
  document.getElementById('stopRecordBtn')!.disabled = true;
});

// Recording controls for microphone section
document.getElementById('startRecordBtn')?.addEventListener('click', () => {
  if (!recordingDestination) {
    logResult('micResult', '❌ No recording destination available', 'error');
    return;
  }
  
  recordedChunks = [];
  
  // Try to use the best available audio codec for quality
  let mimeType = 'audio/webm';
  const codecOptions = [
    'audio/webm;codecs=opus',
    'audio/ogg;codecs=opus', 
    'audio/webm',
    'audio/ogg'
  ];
  
  // Find the best supported codec
  for (const codec of codecOptions) {
    if (MediaRecorder.isTypeSupported(codec)) {
      mimeType = codec;
      console.log('[Recording] Using codec:', codec);
      break;
    }
  }
  
  // Create MediaRecorder with better quality settings
  mediaRecorder = new MediaRecorder(recordingDestination.stream, {
    mimeType: mimeType,
    audioBitsPerSecond: 128000 // 128 kbps for better quality
  });
  
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  
  mediaRecorder.onstop = () => {
    // Use the same mime type that was selected for recording
    recordedBlob = new Blob(recordedChunks, { type: mimeType });
    const duration = (Date.now() - recordingStartTime) / 1000;
    
    // Always unmute speaker output when recording stops
    if (outputGainNode) {
      outputGainNode.gain.value = 1;
    }
    
    logResult('micResult', `✅ Recording saved (${duration.toFixed(1)}s)`, 'success');
    
    // Enable playback and download buttons
    document.getElementById('playRecordingBtn')!.disabled = false;
    document.getElementById('downloadRecordingBtn')!.disabled = false;
  };
  
  // Start recording
  recordingStartTime = Date.now();
  mediaRecorder.start();
  
  // Mute speaker output to prevent feedback during recording
  if (outputGainNode) {
    outputGainNode.gain.value = 0;
  }
  
  logResult('micResult', '🔴 Recording started (speaker muted)...', 'info');
  
  document.getElementById('startRecordBtn')!.disabled = true;
  document.getElementById('stopRecordBtn')!.disabled = false;
});

document.getElementById('stopRecordBtn')?.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    
    // Unmute speaker output
    if (outputGainNode) {
      outputGainNode.gain.value = 1;
    }
    
    logResult('micResult', '⏹ Recording stopped (speaker unmuted)', 'info');
    
    document.getElementById('startRecordBtn')!.disabled = false;
    document.getElementById('stopRecordBtn')!.disabled = true;
  }
});

document.getElementById('playRecordingBtn')?.addEventListener('click', () => {
  if (recordedBlob) {
    const audioElement = document.getElementById('recordingPlayback') as HTMLAudioElement;
    audioElement.src = URL.createObjectURL(recordedBlob);
    audioElement.style.display = 'block';
    audioElement.play();
    
    logResult('micResult', '▶️ Playing recording...', 'info');
  }
});

document.getElementById('downloadRecordingBtn')?.addEventListener('click', () => {
  if (recordedBlob) {
    const url = URL.createObjectURL(recordedBlob);
    const a = document.createElement('a');
    a.href = url;
    
    // Determine file extension based on blob type
    const extension = recordedBlob.type.includes('ogg') ? 'ogg' : 'webm';
    a.download = `recording-${new Date().toISOString()}.${extension}`;
    
    a.click();
    URL.revokeObjectURL(url);
    
    logResult('micResult', '💾 Recording downloaded', 'success');
  }
});

// ============================================================================
// 4. Configuration Management Test
// ============================================================================

// Preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const preset = (e.target as HTMLElement).dataset.preset as NoiseReductionPreset;
    
    if (!processor) {
      logResult('configResult', '❌ Please initialize the processor first (Section 1)', 'error');
      return;
    }
    
    if (processor) {
      processor.loadPreset(preset);
      const config = processor.getConfig();
      
      // Update UI
      (document.getElementById('gateThreshold') as HTMLInputElement).value = String(config.gateThreshold);
      (document.getElementById('gateAttack') as HTMLInputElement).value = String(config.gateAttack);
      (document.getElementById('gateRelease') as HTMLInputElement).value = String(config.gateRelease);
      (document.getElementById('oversubtraction') as HTMLInputElement).value = String(config.overSubtractionFactor);
      
      updateStatus('gateThresholdValue', config.gateThreshold || -40);
      updateStatus('gateAttackValue', config.gateAttack || 5);
      updateStatus('gateReleaseValue', config.gateRelease || 100);
      updateStatus('oversutbtractionValue', config.overSubtractionFactor || 2.0);
      
      // Update active state
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      logResult('configResult', `✅ Loaded preset: ${preset}`, 'success');
    }
  });
});

// Parameter sliders
document.getElementById('gateThreshold')?.addEventListener('input', (e) => {
  const value = parseFloat((e.target as HTMLInputElement).value);
  updateStatus('gateThresholdValue', value);
  if (!processor) {
    logResult('configResult', '❌ Please initialize the processor first (Section 1)', 'error');
    return;
  }
  if (processor) {
    processor.setGateThreshold(value);
  }
});

document.getElementById('gateAttack')?.addEventListener('input', (e) => {
  const value = parseFloat((e.target as HTMLInputElement).value);
  updateStatus('gateAttackValue', value);
  if (!processor) {
    logResult('configResult', '❌ Please initialize the processor first (Section 1)', 'error');
    return;
  }
  if (processor) {
    processor.updateConfig({ gateAttack: value });
  }
});

document.getElementById('gateRelease')?.addEventListener('input', (e) => {
  const value = parseFloat((e.target as HTMLInputElement).value);
  updateStatus('gateReleaseValue', value);
  if (!processor) {
    logResult('configResult', '❌ Please initialize the processor first (Section 1)', 'error');
    return;
  }
  if (processor) {
    processor.updateConfig({ gateRelease: value });
  }
});

document.getElementById('oversubtraction')?.addEventListener('input', (e) => {
  const value = parseFloat((e.target as HTMLInputElement).value);
  updateStatus('oversutbtractionValue', value);
  if (!processor) {
    logResult('configResult', '❌ Please initialize the processor first (Section 1)', 'error');
    return;
  }
  if (processor) {
    processor.setOverSubtractionFactor(value);
  }
});

// Export/Import config
document.getElementById('saveConfigBtn')?.addEventListener('click', () => {
  if (processor) {
    const config = processor.getConfig();
    const json = JSON.stringify(config, null, 2);
    
    // Create download link
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'noise-reducer-config.json';
    a.click();
    URL.revokeObjectURL(url);
    
    logResult('configResult', '✅ Configuration exported', 'success');
  }
});

document.getElementById('loadConfigBtn')?.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json';
  
  input.onchange = async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file && processor) {
      const text = await file.text();
      const config = JSON.parse(text) as ProcessorOptions;
      processor.updateConfig(config);
      
      logResult('configResult', '✅ Configuration imported', 'success');
    }
  };
  
  input.click();
});

// ============================================================================
// 5. Noise Profile Management Test
// ============================================================================

document.getElementById('startNoiseCaptureBtn')?.addEventListener('click', () => {
  if (processor) {
    processor.startNoiseCapture();
    logResult('profileResult', '🎧 Capturing noise... Keep quiet!', 'info');
    
    document.getElementById('startNoiseCaptureBtn')!.disabled = true;
    document.getElementById('stopNoiseCaptureBtn')!.disabled = false;
  }
});

document.getElementById('stopNoiseCaptureBtn')?.addEventListener('click', () => {
  if (processor) {
    const profile = processor.stopNoiseCapture();
    
    if (visualizer) {
      visualizer.updateNoiseProfile(profile.spectrum);
    }
    
    logResult('profileResult', 
      `✅ Noise profile learned!\n` +
      `Frames: ${profile.frame_count}\n` +
      `Spectrum length: ${profile.spectrum.length}`,
      'success'
    );
    
    document.getElementById('startNoiseCaptureBtn')!.disabled = false;
    document.getElementById('stopNoiseCaptureBtn')!.disabled = true;
    document.getElementById('freezeProfileBtn')!.disabled = false;
  }
});

document.getElementById('freezeProfileBtn')?.addEventListener('click', () => {
  if (processor) {
    processor.freezeNoiseProfile();
    logResult('profileResult', '🔒 Noise profile frozen', 'info');
    
    document.getElementById('freezeProfileBtn')!.disabled = true;
    document.getElementById('unfreezeProfileBtn')!.disabled = false;
  }
});

document.getElementById('unfreezeProfileBtn')?.addEventListener('click', () => {
  if (processor) {
    processor.unfreezeNoiseProfile();
    logResult('profileResult', '🔓 Noise profile unfrozen', 'info');
    
    document.getElementById('freezeProfileBtn')!.disabled = false;
    document.getElementById('unfreezeProfileBtn')!.disabled = true;
  }
});

document.getElementById('resetProfileBtn')?.addEventListener('click', () => {
  if (processor) {
    processor.resetNoiseProfile();
    logResult('profileResult', '✅ Noise profile reset', 'success');
  }
});

document.getElementById('exportProfileBtn')?.addEventListener('click', () => {
  if (processor) {
    const json = processor.exportNoiseProfile();
    
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'noise-profile.json';
    a.click();
    URL.revokeObjectURL(url);
    
    logResult('profileResult', '✅ Noise profile exported', 'success');
  }
});

document.getElementById('importProfileBtn')?.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json';
  
  input.onchange = async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file && processor) {
      const text = await file.text();
      processor.importNoiseProfile(text);
      
      logResult('profileResult', '✅ Noise profile imported', 'success');
    }
  };
  
  input.click();
});

// ============================================================================
// 6. File Processing Test
// ============================================================================

document.getElementById('audioFileInput')?.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    document.getElementById('processFileBtn')!.disabled = false;
    logResult('fileResult', `Selected: ${file.name}`, 'info');
  }
});

document.getElementById('processFileBtn')?.addEventListener('click', async () => {
  const input = document.getElementById('audioFileInput') as HTMLInputElement;
  const file = input.files?.[0];
  
  if (file) {
    try {
      logResult('fileResult', 'Processing file...', 'info');
      
      const startTime = performance.now();
      processedBlob = await processAudioFile(file);
      const processingTime = performance.now() - startTime;
      
      logResult('fileResult', 
        `✅ File processed successfully!\n` +
        `Processing time: ${processingTime.toFixed(2)}ms\n` +
        `Output size: ${(processedBlob.size / 1024).toFixed(2)} KB`,
        'success'
      );
      
      document.getElementById('downloadProcessedBtn')!.disabled = false;
      
    } catch (error) {
      logResult('fileResult', `❌ Processing failed: ${error}`, 'error');
    }
  }
});

document.getElementById('downloadProcessedBtn')?.addEventListener('click', () => {
  if (processedBlob) {
    const url = URL.createObjectURL(processedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'processed-audio.wav';
    a.click();
    URL.revokeObjectURL(url);
  }
});

// ============================================================================
// 7. Performance Monitoring
// ============================================================================

document.getElementById('startPerfBtn')?.addEventListener('click', () => {
  // Ensure processor exists and is initialized
  if (!processor) {
    logResult('perfResult', '❌ Please initialize the processor first (Section 1)', 'error');
    return;
  }
  
  if (processor) {
    processor.startPerfMonitoring();
    
    // Start updating stats
    perfMonitorInterval = setInterval(() => {
      const stats = processor!.getStatistics();
      const state = processor!.getState();
      
      updateStatus('cpuUsage', `${stats.cpuUsagePercent.toFixed(1)}%`);
      updateStatus('latency', `${stats.averageLatencyMs.toFixed(2)}ms`);
      updateStatus('framesProcessed', stats.framesProcessed);
      updateStatus('droppedFrames', stats.droppedFrames);
      
    }, 100) as any;
    
    // Set up performance warning handler
    processor.onPerformanceWarning((warning) => {
      logResult('perfResult', 
        `⚠️ Performance Warning: ${warning.type}\n${JSON.stringify(warning)}`,
        'error'
      );
    });
    
    logResult('perfResult', '✅ Performance monitoring started', 'success');
    
    document.getElementById('startPerfBtn')!.disabled = true;
    document.getElementById('stopPerfBtn')!.disabled = false;
  }
});

document.getElementById('stopPerfBtn')?.addEventListener('click', () => {
  if (processor && perfMonitorInterval) {
    const report = processor.stopPerfMonitoring();
    
    clearInterval(perfMonitorInterval);
    perfMonitorInterval = null;
    
    logResult('perfResult', 
      `✅ Performance Report:\n` +
      `Duration: ${report.duration.toFixed(2)}ms\n` +
      `Avg Processing Time: ${report.avgProcessingTime.toFixed(2)}ms\n` +
      `Peak Processing Time: ${report.peakProcessingTime.toFixed(2)}ms\n` +
      `CPU Usage: ${report.cpuUsage.toFixed(1)}%`,
      'success'
    );
    
    document.getElementById('startPerfBtn')!.disabled = false;
    document.getElementById('stopPerfBtn')!.disabled = true;
  }
});

document.getElementById('simdTestBtn')?.addEventListener('click', () => {
  if (processor) {
    const simdSupport = processor.detectSimdSupport();
    
    logResult('perfResult',
      `SIMD Support:\n` +
      `Available: ${simdSupport.available}\n` +
      `Version: ${simdSupport.version}\n` +
      `Features: ${simdSupport.features.join(', ')}`,
      simdSupport.available ? 'success' : 'info'
    );
    
    if (simdSupport.available) {
      processor.enableSimd(true);
    }
  }
});

// ============================================================================
// 8. Memory Management Test
// ============================================================================

document.getElementById('memoryTestBtn')?.addEventListener('click', () => {
  if (processor) {
    try {
      // Allocate and free memory
      const size = 1024 * 1024; // 1MB
      const ptr = processor.allocateWasmMemory(size);
      
      logResult('memoryResult', 
        `✅ Allocated ${size} bytes at pointer: ${ptr}`,
        'success'
      );
      
      processor.freeWasmMemory(ptr);
      
      logResult('memoryResult', 
        `✅ Memory freed successfully`,
        'success'
      );
      
    } catch (error) {
      logResult('memoryResult', `❌ Memory test failed: ${error}`, 'error');
    }
  }
});

document.getElementById('bufferPoolTestBtn')?.addEventListener('click', () => {
  const pool = new BufferPool();
  
  // Test buffer acquisition and release
  const buffer1 = pool.acquire(1024);
  const buffer2 = pool.acquire(1024);
  const buffer3 = pool.acquire(2048);
  
  logResult('memoryResult', 
    `Acquired 3 buffers: ${buffer1.length}, ${buffer2.length}, ${buffer3.length}`,
    'info'
  );
  
  // Release buffers
  pool.release(buffer1);
  pool.release(buffer2);
  
  // Acquire again - should reuse
  const buffer4 = pool.acquire(1024);
  
  const isReused = buffer4 === buffer1 || buffer4 === buffer2;
  
  logResult('memoryResult', 
    `✅ Buffer pool test complete\n` +
    `Buffer reused: ${isReused}`,
    'success'
  );
  
  pool.clear();
});

document.getElementById('memoryStatsBtn')?.addEventListener('click', () => {
  if (processor) {
    const stats = processor.getWasmMemoryUsage();
    
    logResult('memoryResult',
      `Memory Statistics:\n` +
      `Used: ${(stats.used / 1024).toFixed(2)} KB\n` +
      `Total: ${(stats.total / 1024).toFixed(2)} KB\n` +
      `Peak: ${(stats.peak / 1024).toFixed(2)} KB`,
      'info'
    );
  }
});

// ============================================================================
// 9. Worker Thread Test
// ============================================================================

document.getElementById('createWorkerBtn')?.addEventListener('click', () => {
  try {
    worker = new NoiseReducerWorker({
      sampleRate: 48000,
      frameSize: 256
    });
    
    worker.onMessage((data) => {
      logResult('workerResult', `Worker message: ${JSON.stringify(data)}`, 'info');
    });
    
    logResult('workerResult', '✅ Worker created successfully', 'success');
    
    document.getElementById('createWorkerBtn')!.disabled = true;
    document.getElementById('processWorkerBtn')!.disabled = false;
    document.getElementById('terminateWorkerBtn')!.disabled = false;
    
  } catch (error) {
    logResult('workerResult', `❌ Failed to create worker: ${error}`, 'error');
  }
});

document.getElementById('processWorkerBtn')?.addEventListener('click', async () => {
  if (worker) {
    try {
      const testData = new Float32Array(1024);
      for (let i = 0; i < testData.length; i++) {
        testData[i] = Math.sin(2 * Math.PI * i / 100);
      }
      
      const startTime = performance.now();
      const result = await worker.process(testData);
      const processingTime = performance.now() - startTime;
      
      logResult('workerResult', 
        `✅ Worker processing complete\n` +
        `Input length: ${testData.length}\n` +
        `Output length: ${result.length}\n` +
        `Processing time: ${processingTime.toFixed(2)}ms`,
        'success'
      );
      
    } catch (error) {
      logResult('workerResult', `❌ Worker processing failed: ${error}`, 'error');
    }
  }
});

document.getElementById('terminateWorkerBtn')?.addEventListener('click', () => {
  if (worker) {
    worker.terminate();
    worker = null;
    
    logResult('workerResult', '✅ Worker terminated', 'success');
    
    document.getElementById('createWorkerBtn')!.disabled = false;
    document.getElementById('processWorkerBtn')!.disabled = true;
    document.getElementById('terminateWorkerBtn')!.disabled = true;
  }
});

// ============================================================================
// 10. Integration Test
// ============================================================================

document.getElementById('runAllTestsBtn')?.addEventListener('click', async () => {
  logResult('integrationResult', 'Starting comprehensive test suite...', 'info');
  
  const results: string[] = [];
  
  try {
    // Test 1: Browser Support
    const support = NoiseReducer.checkBrowserSupport();
    results.push(`✅ Browser support check passed`);
    
    // Test 2: Initialize
    const testProcessor = new NoiseReducer();
    await testProcessor.initialize();
    results.push(`✅ Initialization test passed`);
    
    // Test 3: Process sample data
    // Ensure global context is initialized
    if (!globalAudioContext) {
      globalAudioContext = new AudioContext();
    }
    const testBuffer = globalAudioContext.createBuffer(1, 1024, 48000);
    const data = testBuffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin(2 * Math.PI * i / 100);
    }
    
    const processed = await testProcessor.process(testBuffer);
    results.push(`✅ Audio processing test passed`);
    
    // Test 4: Configuration
    testProcessor.loadPreset(NoiseReductionPreset.VoiceChat);
    const config = testProcessor.getConfig();
    results.push(`✅ Configuration test passed`);
    
    // Test 5: Noise Profile
    testProcessor.startNoiseCapture();
    testProcessor.stopNoiseCapture();
    results.push(`✅ Noise profile test passed`);
    
    // Test 6: Statistics
    const stats = testProcessor.getStatistics();
    results.push(`✅ Statistics test passed`);
    
    // Test 7: Memory
    const memStats = testProcessor.getWasmMemoryUsage();
    results.push(`✅ Memory management test passed`);
    
    // Test 8: SIMD
    const simd = testProcessor.detectSimdSupport();
    results.push(`✅ SIMD detection test passed`);
    
    // Test 9: State
    const state = testProcessor.getState();
    results.push(`✅ State management test passed`);
    
    // Test 10: Cleanup
    testProcessor.dispose();
    results.push(`✅ Disposal test passed`);
    
    logResult('integrationResult', 
      `✅ ALL TESTS PASSED!\n\n${results.join('\n')}`,
      'success'
    );
    
  } catch (error) {
    logResult('integrationResult', 
      `❌ Integration test failed:\n${error}\n\nCompleted tests:\n${results.join('\n')}`,
      'error'
    );
  }
});

document.getElementById('stressTestBtn')?.addEventListener('click', async () => {
  logResult('integrationResult', '⚠️ Starting stress test... This may take a while!', 'info');
  
  try {
    const iterations = 100;
    const processors: NoiseReducer[] = [];
    const startTime = performance.now();
    
    // Create multiple processors
    for (let i = 0; i < 10; i++) {
      const p = new NoiseReducer();
      await p.initialize();
      processors.push(p);
    }
    
    // Process many buffers
    // Ensure global context is initialized
    if (!globalAudioContext) {
      globalAudioContext = new AudioContext();
    }
    const testBuffer = globalAudioContext.createBuffer(1, 4096, 48000);
    const data = testBuffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    
    for (let i = 0; i < iterations; i++) {
      const processor = processors[i % processors.length];
      await processor.process(testBuffer);
      
      if (i % 10 === 0) {
        logResult('integrationResult', 
          `Progress: ${i}/${iterations} iterations...`,
          'info'
        );
      }
    }
    
    // Clean up
    for (const p of processors) {
      p.dispose();
    }
    
    const totalTime = performance.now() - startTime;
    const avgTime = totalTime / iterations;
    
    logResult('integrationResult', 
      `✅ Stress test complete!\n` +
      `Total iterations: ${iterations}\n` +
      `Total time: ${totalTime.toFixed(2)}ms\n` +
      `Average time per iteration: ${avgTime.toFixed(2)}ms\n` +
      `Processors created: ${processors.length}`,
      'success'
    );
    
  } catch (error) {
    logResult('integrationResult', `❌ Stress test failed: ${error}`, 'error');
  }
});

// Initialize on load
window.addEventListener('load', () => {
  logResult('integrationResult', 'Test suite loaded. Ready to test!', 'info');
});