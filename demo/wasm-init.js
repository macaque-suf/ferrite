// Shared WASM initialization script for all demo pages
export async function initializeWasm() {
    try {
        // Load WASM module dynamically at runtime
        const wasmModuleUrl = new URL('/wasm/wasm_audio_ferrite.js', import.meta.url).href;
        const wasmBinaryUrl = new URL('/wasm/wasm_audio_ferrite_bg.wasm', import.meta.url).href;
        
        // Dynamic import with vite-ignore comment
        const module = await import(/* @vite-ignore */ wasmModuleUrl);
        await module.default(wasmBinaryUrl);
        
        return module;
    } catch (error) {
        console.error('Failed to initialize WASM:', error);
        throw error;
    }
}