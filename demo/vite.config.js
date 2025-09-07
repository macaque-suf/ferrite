import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  publicDir: 'public',
  build: {
    assetsDir: 'assets',
    copyPublicDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        livePassthrough: resolve(__dirname, 'live-passthrough.html'),
        testNoiseReduction: resolve(__dirname, 'test-noise-reduction.html'),
        testZeroCopy: resolve(__dirname, 'test-zero-copy.html')
      }
    }
  },
  server: {
    fs: {
      allow: ['..']
    }
  }
});