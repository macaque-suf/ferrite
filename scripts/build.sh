#!/bin/bash
set -e

echo "🦀 Building Rust/WASM..."
cd audio-processor-core
wasm-pack build --target web --out-dir ../packages/toolkit/src/wasm
cd ..

echo "📦 Building TypeScript..."
cd packages/toolkit
npm run build
cd ../..

echo "✅ Build complete!"
