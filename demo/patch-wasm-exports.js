#!/usr/bin/env node
// This script patches the wasm-bindgen generated JS to expose the memory

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const wasmJsPath = path.join(__dirname, 'dist/wasm/wasm_audio_ferrite.js');

if (!fs.existsSync(wasmJsPath)) {
    console.log('WASM JS file not found, skipping patch');
    process.exit(0);
}

let content = fs.readFileSync(wasmJsPath, 'utf8');

// Add a function to get the memory after the wasm variable is defined
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

// Find the location after the exports
const exportDefaultIndex = content.indexOf('export default __wbg_init;');
if (exportDefaultIndex !== -1) {
    // Insert our patch before the default export
    content = content.slice(0, exportDefaultIndex) + patchCode + '\n' + content.slice(exportDefaultIndex);
    
    fs.writeFileSync(wasmJsPath, content);
    console.log('✅ Patched WASM exports to include memory access functions');
} else {
    console.error('Could not find export location in WASM JS file');
    process.exit(1);
}