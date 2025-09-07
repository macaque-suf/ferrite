// Shared WASM initialization script for all demo pages
export async function initializeWasm() {
    try {
        // Load WASM module dynamically at runtime
        const wasmModuleUrl = new URL('/wasm/wasm_audio_ferrite.js', import.meta.url).href;
        const wasmBinaryUrl = new URL('/wasm/wasm_audio_ferrite_bg.wasm', import.meta.url).href;
        
        // Dynamic import with vite-ignore comment
        const module = await import(/* @vite-ignore */ wasmModuleUrl);
        
        // Initialize and get the actual WASM instance
        const wasmInstance = await module.default(wasmBinaryUrl);
        
        // The memory is exported as __wbindgen_export_0 by wasm-bindgen
        // This is the WebAssembly.Memory object we need for zero-copy
        if (module.__wbindgen_export_0) {
            module.memory = module.__wbindgen_export_0;
            console.log('WASM memory found in __wbindgen_export_0');
        }
        
        // Also check if the init function exposes the wasm module
        if (module.default && module.default.__wbindgen_wasm_module) {
            const wasmModule = module.default.__wbindgen_wasm_module;
            if (wasmModule && wasmModule.exports && wasmModule.exports.memory) {
                module.memory = wasmModule.exports.memory;
                console.log('WASM memory found in module.exports.memory');
            }
        }
        
        // Store the instance reference if available
        module.instance = wasmInstance;
        
        return module;
    } catch (error) {
        console.error('Failed to initialize WASM:', error);
        throw error;
    }
}