let audioContext: AudioContext | null = null;

export function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
  return audioContext;
}

export function createAudioBuffer(
  data: Float32Array,
  sampleRate: number = 48000
): AudioBuffer {
  const ctx = getAudioContext();
  const buffer = ctx.createBuffer(1, data.length, sampleRate);
  // Create a new Float32Array to ensure correct type
  const channelData = new Float32Array(data);
  buffer.copyToChannel(channelData, 0);
  return buffer;
}
