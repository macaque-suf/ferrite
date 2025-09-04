import { NoiseReducer } from '../packages/toolkit/src';
import { SpectrumVisualizer, GateState } from './visualizer';

let audioContext: AudioContext;
let microphone: MediaStreamAudioSourceNode;
let processor: NoiseReducer;
let processingEnabled = false;
let visualizer: SpectrumVisualizer;

const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const toggleBtn = document.getElementById('toggleBtn') as HTMLButtonElement;
const status = document.getElementById('status') as HTMLDivElement;
const meterBar = document.getElementById('meterBar') as HTMLDivElement;

// Visualizer control buttons
const toggleScaleBtn = document.getElementById('toggleScale') as HTMLButtonElement;
const togglePeaksBtn = document.getElementById('togglePeaks') as HTMLButtonElement;
const toggleGridBtn = document.getElementById('toggleGrid') as HTMLButtonElement;
const learnNoiseBtn = document.getElementById('learnNoise') as HTMLButtonElement;

startBtn.addEventListener('click', async () => {
  try {
    status.textContent = 'Requesting microphone access...';
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      } 
    });
    
    audioContext = new AudioContext();
    microphone = audioContext.createMediaStreamSource(stream);
    
    // Initialize processor
    processor = new NoiseReducer({ 
      sampleRate: audioContext.sampleRate 
    });
    await processor.initialize();
    
    // Set up audio processing chain
    const analyser = audioContext.createAnalyser();
    microphone.connect(analyser);
    
    // Initialize and connect visualizer
    visualizer = new SpectrumVisualizer({
      canvasId: 'spectrumCanvas',
      fftSize: 2048,
      smoothingTimeConstant: 0.75,
      minDecibels: -90,
      maxDecibels: -10
    });
    
    await visualizer.initialize();
    visualizer.connectAudioNode(microphone);
    visualizer.start();
    
    // Visual feedback
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    function updateMeter() {
      analyser.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      meterBar.style.width = `${(average / 255) * 100}%`;
      requestAnimationFrame(updateMeter);
    }
    updateMeter();
    
    status.textContent = 'Microphone active. Noise reduction ready.';
    startBtn.disabled = true;
    toggleBtn.disabled = false;
    learnNoiseBtn.disabled = false;
    
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  }
});

toggleBtn.addEventListener('click', () => {
  processingEnabled = !processingEnabled;
  toggleBtn.textContent = processingEnabled 
    ? 'Disable Noise Reduction' 
    : 'Enable Noise Reduction';
  
  status.textContent = processingEnabled
    ? '🟢 Noise reduction active'
    : '⭕ Noise reduction disabled';
  
  // Update visualizer gate state
  if (visualizer) {
    visualizer.updateGateState(processingEnabled ? GateState.Open : GateState.Closed);
  }
});

// Visualizer control event listeners
let scaleMode: 'log' | 'linear' = 'log';
toggleScaleBtn?.addEventListener('click', () => {
  scaleMode = scaleMode === 'log' ? 'linear' : 'log';
  toggleScaleBtn.textContent = `${scaleMode === 'log' ? 'Log' : 'Linear'} Scale`;
  visualizer?.setFrequencyScale(scaleMode);
});

let peaksEnabled = true;
togglePeaksBtn?.addEventListener('click', () => {
  peaksEnabled = !peaksEnabled;
  togglePeaksBtn.textContent = `Peaks: ${peaksEnabled ? 'ON' : 'OFF'}`;
  visualizer?.setShowPeaks(peaksEnabled);
});

let gridEnabled = true;
toggleGridBtn?.addEventListener('click', () => {
  gridEnabled = !gridEnabled;
  toggleGridBtn.textContent = `Grid: ${gridEnabled ? 'ON' : 'OFF'}`;
  visualizer?.setShowGrid(gridEnabled);
});

// Learn noise profile
let noiseProfileLearned = false;
learnNoiseBtn?.addEventListener('click', () => {
  if (!visualizer || !audioContext) {
    status.textContent = '⚠️ Start microphone first';
    return;
  }
  
  status.textContent = '🎧 Learning noise profile (keep quiet for 2 seconds)...';
  learnNoiseBtn.disabled = true;
  
  // Simulate learning noise profile
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  microphone.connect(analyser);
  
  const bufferLength = analyser.frequencyBinCount;
  const noiseProfile = new Float32Array(bufferLength);
  const samples: Float32Array[] = [];
  
  // Collect samples over 2 seconds
  const sampleInterval = setInterval(() => {
    const dataArray = new Float32Array(bufferLength);
    analyser.getFloatFrequencyData(dataArray);
    samples.push(new Float32Array(dataArray));
  }, 100);
  
  setTimeout(() => {
    clearInterval(sampleInterval);
    
    // Average the samples to create noise profile
    for (let i = 0; i < bufferLength; i++) {
      let sum = 0;
      for (const sample of samples) {
        sum += sample[i];
      }
      noiseProfile[i] = sum / samples.length;
    }
    
    // Update visualizer with noise profile
    visualizer.updateNoiseProfile(noiseProfile);
    
    status.textContent = '✅ Noise profile learned successfully';
    learnNoiseBtn.disabled = false;
    learnNoiseBtn.textContent = 'Update Noise Profile';
    noiseProfileLearned = true;
    
    // Cleanup
    analyser.disconnect();
  }, 2000);
});
