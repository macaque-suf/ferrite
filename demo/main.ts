import { NoiseReducer } from '../packages/toolkit/src';

let audioContext: AudioContext;
let microphone: MediaStreamAudioSourceNode;
let processor: NoiseReducer;
let processingEnabled = false;

const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const toggleBtn = document.getElementById('toggleBtn') as HTMLButtonElement;
const status = document.getElementById('status') as HTMLDivElement;
const meterBar = document.getElementById('meterBar') as HTMLDivElement;

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
});
