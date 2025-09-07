// Load WASM module dynamically at runtime
export async function loadWasmModule() {
  // Load the WASM module from the public directory at runtime
  // This avoids Vite trying to bundle it during build
  const wasmModuleUrl = new URL('/wasm/wasm_audio_ferrite.js', import.meta.url).href;
  const wasmBinaryUrl = new URL('/wasm/wasm_audio_ferrite_bg.wasm', import.meta.url).href;
  
  // Dynamically import the module
  const module = await import(/* @vite-ignore */ wasmModuleUrl);
  
  // Initialize with the WASM binary
  await module.default(wasmBinaryUrl);
  
  return module;
}