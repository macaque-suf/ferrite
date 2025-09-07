import { defineConfig } from 'vite';

export default defineConfig({
  publicDir: 'public',
  build: {
    assetsDir: 'assets',
    copyPublicDir: true
  },
  server: {
    fs: {
      allow: ['..']
    }
  }
});