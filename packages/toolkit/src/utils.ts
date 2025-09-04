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
  buffer.copyToChannel(data, 0);
  return buffer;
}
