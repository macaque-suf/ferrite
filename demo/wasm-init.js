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
        
        // Create a wrapper object that includes all exports plus memory
        // This avoids the "object is not extensible" error
        const wasmWrapper = {
            ...module,  // Spread all original exports
            instance: wasmInstance
        };
        
        // Use our patched export function to get memory if available
        if (module.getWasmMemory) {
            wasmWrapper.memory = module.getWasmMemory();
            console.log('WASM memory obtained via getWasmMemory()');
        }
        // Fallback to __wbindgen_export_0 if patch wasn't applied
        else if (module.__wbindgen_export_0) {
            wasmWrapper.memory = module.__wbindgen_export_0;
            console.log('WASM memory found in __wbindgen_export_0');
        }
        // Also check if the init function exposes the wasm module
        else if (module.default && module.default.__wbindgen_wasm_module) {
            const wasmModule = module.default.__wbindgen_wasm_module;
            if (wasmModule && wasmModule.exports && wasmModule.exports.memory) {
                wasmWrapper.memory = wasmModule.exports.memory;
                console.log('WASM memory found in module.exports.memory');
            }
        }
        
        if (!wasmWrapper.memory) {
            console.warn('Could not find WASM memory - zero-copy operations will not work');
        }
        
        return wasmWrapper;
    } catch (error) {
        console.error('Failed to initialize WASM:', error);
        throw error;
    }
}