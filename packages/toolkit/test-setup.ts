// Test setup file for vitest
// Provides polyfills and global test utilities

// Polyfill for TextEncoder/TextDecoder if needed
import { TextEncoder, TextDecoder } from 'util';
Object.assign(global, { TextEncoder, TextDecoder });

// Mock MediaStream if not available
if (typeof MediaStream === 'undefined') {
  global.MediaStream = class MediaStream {
    private tracks: any[] = [];
    
    getTracks() {
      return this.tracks;
    }
    
    addTrack(track: any) {
      this.tracks.push(track);
    }
    
    removeTrack(track: any) {
      const index = this.tracks.indexOf(track);
      if (index > -1) {
        this.tracks.splice(index, 1);
      }
    }
  } as any;
}

// Add custom matchers if needed
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});