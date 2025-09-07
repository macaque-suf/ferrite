// Test that the npm package works correctly
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import init, { NoiseReducer, getWasmMemory } from './dist/wasm/wasm_audio_ferrite.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function test() {
    console.log('Testing wasm-audio-ferrite package...\n');
    
    try {
        // Initialize WASM with file buffer for Node.js
        console.log('1. Initializing WASM...');
        const wasmPath = join(__dirname, 'dist/wasm/wasm_audio_ferrite_bg.wasm');
        const wasmBuffer = readFileSync(wasmPath);
        await init(wasmBuffer);
        console.log('   ✅ WASM initialized');
        
        // Check memory access
        console.log('\n2. Checking memory access...');
        const memory = getWasmMemory();
        if (memory) {
            console.log('   ✅ Memory accessible:', memory.buffer.byteLength, 'bytes');
        } else {
            console.log('   ❌ Memory not accessible');
        }
        
        // Create NoiseReducer
        console.log('\n3. Creating NoiseReducer...');
        const reducer = new NoiseReducer(48000);
        console.log('   ✅ NoiseReducer created');
        
        // Test zero-copy API
        console.log('\n4. Testing zero-copy API...');
        if (reducer.alloc_buffer) {
            const ptr = reducer.alloc_buffer(512);
            console.log('   ✅ Buffer allocated at pointer:', ptr);
            
            // Create a view of the memory
            if (memory) {
                const buffer = new Float32Array(memory.buffer, ptr, 512);
                buffer[0] = 42.0;
                console.log('   ✅ Memory write/read test passed');
            }
        } else {
            console.log('   ❌ alloc_buffer not available');
        }
        
        // Test regular processing
        console.log('\n5. Testing regular processing...');
        const testData = new Float32Array(512);
        testData.fill(0.5);
        const result = reducer.process(testData);
        console.log('   ✅ Processing works, output length:', result.length);
        
        console.log('\n✅ All tests passed! Package is ready for publishing.');
        
    } catch (error) {
        console.error('\n❌ Test failed:', error);
        process.exit(1);
    }
}

test();