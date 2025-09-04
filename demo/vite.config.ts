import { defineConfig } from 'vite';
import wasm from '@rollup/plugin-wasm';

export default defineConfig({
  plugins: [wasm()],
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
});
