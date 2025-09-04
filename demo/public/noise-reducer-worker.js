// Noise Reducer Worker
// This worker handles audio processing in a separate thread

self.onmessage = function(event) {
  const { id, type, buffer } = event.data;
  
  switch(type) {
    case 'process':
      // For now, just pass through the audio
      // In production, this would load and use the WASM module
      self.postMessage({
        id,
        type: 'processed',
        data: buffer
      });
      break;
      
    case 'init':
      self.postMessage({
        id,
        type: 'initialized',
        data: true
      });
      break;
      
    default:
      self.postMessage({
        id,
        type: 'error',
        error: 'Unknown message type: ' + type
      });
  }
};