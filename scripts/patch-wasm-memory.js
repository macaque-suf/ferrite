#!/usr/bin/env node
// Patches the wasm-bindgen generated JS to expose memory for zero-copy operations

const fs = require('fs');
const path = require('path');

const wasmJsPath = path.join(__dirname, '../dist/wasm/wasm_audio_ferrite.js');

if (!fs.existsSync(wasmJsPath)) {
    console.log('WASM JS file not found at', wasmJsPath);
    process.exit(1);
}

let content = fs.readFileSync(wasmJsPath, 'utf8');

// Check if already patched
if (content.includes('export function getWasmMemory()')) {
    console.log('✅ WASM exports already patched');
    process.exit(0);
}

// Add functions to access memory after the wasm variable is defined
const patchCode = `
// Patched: Export function to access WASM memory for zero-copy operations
export function getWasmMemory() {
    if (wasm && wasm.memory) {
        return wasm.memory;
    }
    return null;
}

// Patched: Export the raw wasm exports for debugging
export function getWasmExports() {
    return wasm;
}
`;

// Find the location before the default export
const exportDefaultIndex = content.indexOf('export default __wbg_init;');
if (exportDefaultIndex !== -1) {
    // Insert our patch before the default export
    content = content.slice(0, exportDefaultIndex) + patchCode + '\n' + content.slice(exportDefaultIndex);
    
    fs.writeFileSync(wasmJsPath, content);
    console.log('✅ Patched WASM exports to include memory access functions');
    
    // Also patch the TypeScript definitions
    const wasmDtsPath = wasmJsPath.replace('.js', '.d.ts');
    if (fs.existsSync(wasmDtsPath)) {
        let dtsContent = fs.readFileSync(wasmDtsPath, 'utf8');
        
        // Check if already patched
        if (!dtsContent.includes('export function getWasmMemory()')) {
            const dtsPatc = `/**
* Get the WebAssembly memory object for zero-copy operations
* Returns null if WASM is not initialized
*/
export function getWasmMemory(): WebAssembly.Memory | null;

/**
* Get the raw WASM exports for debugging
* Returns undefined if WASM is not initialized
*/
export function getWasmExports(): any;

`;
            // Find a good place to insert (before SyncInitInput)
            const syncInitIndex = dtsContent.indexOf('export type SyncInitInput');
            if (syncInitIndex !== -1) {
                dtsContent = dtsContent.slice(0, syncInitIndex) + dtsPatc + dtsContent.slice(syncInitIndex);
                fs.writeFileSync(wasmDtsPath, dtsContent);
                console.log('✅ Patched TypeScript definitions');
            }
        }
    }
} else {
    console.error('❌ Could not find export location in WASM JS file');
    process.exit(1);
}