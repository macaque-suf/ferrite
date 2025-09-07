let wasm;

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let WASM_VECTOR_LEN = 0;

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}
/**
 * Example greeting function for testing WASM bindings
 * @param {string} name
 * @returns {string}
 */
export function greet(name) {
    let deferred2_0;
    let deferred2_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export_0, wasm.__wbindgen_export_2);
        const len0 = WASM_VECTOR_LEN;
        wasm.greet(retptr, ptr0, len0);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred2_0 = r0;
        deferred2_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export_1(deferred2_0, deferred2_1, 1);
    }
}

const NoiseReducerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_noisereducer_free(ptr >>> 0, 1));
/**
 * WebAssembly-compatible noise reduction processor
 * Combines noise gate and spectral subtraction for comprehensive noise reduction
 */
export class NoiseReducer {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NoiseReducerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_noisereducer_free(ptr, 0);
    }
    /**
     * Returns the frame size used for processing
     * @returns {number}
     */
    get frame_size() {
        const ret = wasm.noisereducer_frame_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Enables or disables bypass mode
     * @param {boolean} bypass
     */
    set_bypass(bypass) {
        wasm.noisereducer_set_bypass(this.__wbg_ptr, bypass);
    }
    /**
     * Get the size of the last allocated buffer
     * @returns {number}
     */
    get buffer_size() {
        const ret = wasm.noisereducer_buffer_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Learns noise profile from a buffer of noise-only samples
     * @param {Float32Array} noise_samples
     */
    learn_noise(noise_samples) {
        const ptr0 = passArrayF32ToWasm0(noise_samples, wasm.__wbindgen_export_0);
        const len0 = WASM_VECTOR_LEN;
        wasm.noisereducer_learn_noise(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Process audio in-place using a buffer in WASM memory.
     * This avoids allocation and copying across the WASM boundary.
     *
     * # Safety
     * The caller must ensure the pointer points to valid memory of at least `len` floats.
     * @param {number} ptr
     * @param {number} len
     */
    processPtr(ptr, len) {
        wasm.noisereducer_processPtr(this.__wbg_ptr, ptr, len);
    }
    /**
     * Returns the current sample rate
     * @returns {number}
     */
    get sample_rate() {
        const ret = wasm.noisereducer_sample_rate(this.__wbg_ptr);
        return ret;
    }
    /**
     * Allocate a buffer in WASM memory and return its pointer.
     * The caller can create a Float32Array view over this memory in JS.
     *
     * # Safety
     * The returned pointer is valid until the next call to alloc_buffer or
     * until the NoiseReducer is dropped.
     * @param {number} len
     * @returns {number}
     */
    alloc_buffer(len) {
        const ret = wasm.noisereducer_alloc_buffer(this.__wbg_ptr, len);
        return ret >>> 0;
    }
    /**
     * Process audio from input buffer to output buffer (both in WASM memory).
     * This allows processing without in-place constraints.
     *
     * # Safety
     * The caller must ensure both pointers point to valid memory of at least `len` floats.
     * The input and output buffers must not overlap.
     * @param {number} in_ptr
     * @param {number} out_ptr
     * @param {number} len
     */
    processInto(in_ptr, out_ptr, len) {
        wasm.noisereducer_processInto(this.__wbg_ptr, in_ptr, out_ptr, len);
    }
    /**
     * Enable or disable the noise gate
     * @param {boolean} enabled
     */
    set_gate_enabled(enabled) {
        wasm.noisereducer_set_gate_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * Set noise gate threshold in dB
     * @param {number} threshold_db
     */
    set_gate_threshold(threshold_db) {
        wasm.noisereducer_set_gate_threshold(this.__wbg_ptr, threshold_db);
    }
    /**
     * Sets the noise reduction aggressiveness (0.0 = minimal, 1.0 = maximum)
     * @param {number} amount
     */
    set_reduction_amount(amount) {
        wasm.noisereducer_set_reduction_amount(this.__wbg_ptr, amount);
    }
    /**
     * Enable or disable spectral subtraction
     * @param {boolean} enabled
     */
    set_spectral_enabled(enabled) {
        wasm.noisereducer_set_spectral_enabled(this.__wbg_ptr, enabled);
    }
    /**
     * Enable or disable Wiener filter mode for spectral subtraction
     * @param {boolean} enabled
     */
    set_wiener_filter_mode(enabled) {
        wasm.noisereducer_set_wiener_filter_mode(this.__wbg_ptr, enabled);
    }
    /**
     * Creates a new noise reducer with the specified sample rate
     * @param {number} sample_rate
     */
    constructor(sample_rate) {
        const ret = wasm.noisereducer_new(sample_rate);
        this.__wbg_ptr = ret >>> 0;
        NoiseReducerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Resets the processor state
     */
    reset() {
        wasm.noisereducer_reset(this.__wbg_ptr);
    }
    /**
     * Processes an audio buffer and returns the denoised output
     * @param {Float32Array} input
     * @returns {Float32Array}
     */
    process(input) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_export_0);
            const len0 = WASM_VECTOR_LEN;
            wasm.noisereducer_process(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export_1(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('wasm_audio_ferrite_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
