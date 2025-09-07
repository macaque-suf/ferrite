/* tslint:disable */
/* eslint-disable */
/**
 * Example greeting function for testing WASM bindings
 */
export function greet(name: string): string;
/**
 * WebAssembly-compatible noise reduction processor
 * Combines noise gate and spectral subtraction for comprehensive noise reduction
 */
export class NoiseReducer {
  free(): void;
  /**
   * Enables or disables bypass mode
   */
  set_bypass(bypass: boolean): void;
  /**
   * Learns noise profile from a buffer of noise-only samples
   */
  learn_noise(noise_samples: Float32Array): void;
  /**
   * Process audio in-place using a buffer in WASM memory.
   * This avoids allocation and copying across the WASM boundary.
   * 
   * # Safety
   * The caller must ensure the pointer points to valid memory of at least `len` floats.
   */
  processPtr(ptr: number, len: number): void;
  /**
   * Allocate a buffer in WASM memory and return its pointer.
   * The caller can create a Float32Array view over this memory in JS.
   * 
   * # Safety
   * The returned pointer is valid until the next call to alloc_buffer or
   * until the NoiseReducer is dropped.
   */
  alloc_buffer(len: number): number;
  /**
   * Process audio from input buffer to output buffer (both in WASM memory).
   * This allows processing without in-place constraints.
   * 
   * # Safety
   * The caller must ensure both pointers point to valid memory of at least `len` floats.
   * The input and output buffers must not overlap.
   */
  processInto(in_ptr: number, out_ptr: number, len: number): void;
  /**
   * Enable or disable the noise gate
   */
  set_gate_enabled(enabled: boolean): void;
  /**
   * Set noise gate threshold in dB
   */
  set_gate_threshold(threshold_db: number): void;
  /**
   * Sets the noise reduction aggressiveness (0.0 = minimal, 1.0 = maximum)
   */
  set_reduction_amount(amount: number): void;
  /**
   * Enable or disable spectral subtraction
   */
  set_spectral_enabled(enabled: boolean): void;
  /**
   * Enable or disable Wiener filter mode for spectral subtraction
   */
  set_wiener_filter_mode(enabled: boolean): void;
  /**
   * Creates a new noise reducer with the specified sample rate
   */
  constructor(sample_rate: number);
  /**
   * Resets the processor state
   */
  reset(): void;
  /**
   * Processes an audio buffer and returns the denoised output
   */
  process(input: Float32Array): Float32Array;
  /**
   * Returns the frame size used for processing
   */
  readonly frame_size: number;
  /**
   * Get the size of the last allocated buffer
   */
  readonly buffer_size: number;
  /**
   * Returns the current sample rate
   */
  readonly sample_rate: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_noisereducer_free: (a: number, b: number) => void;
  readonly greet: (a: number, b: number, c: number) => void;
  readonly noisereducer_alloc_buffer: (a: number, b: number) => number;
  readonly noisereducer_buffer_size: (a: number) => number;
  readonly noisereducer_frame_size: (a: number) => number;
  readonly noisereducer_learn_noise: (a: number, b: number, c: number) => void;
  readonly noisereducer_new: (a: number) => number;
  readonly noisereducer_process: (a: number, b: number, c: number, d: number) => void;
  readonly noisereducer_processInto: (a: number, b: number, c: number, d: number) => void;
  readonly noisereducer_processPtr: (a: number, b: number, c: number) => void;
  readonly noisereducer_reset: (a: number) => void;
  readonly noisereducer_sample_rate: (a: number) => number;
  readonly noisereducer_set_bypass: (a: number, b: number) => void;
  readonly noisereducer_set_gate_enabled: (a: number, b: number) => void;
  readonly noisereducer_set_gate_threshold: (a: number, b: number) => void;
  readonly noisereducer_set_reduction_amount: (a: number, b: number) => void;
  readonly noisereducer_set_spectral_enabled: (a: number, b: number) => void;
  readonly noisereducer_set_wiener_filter_mode: (a: number, b: number) => void;
  readonly __wbindgen_export_0: (a: number, b: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export_1: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export_2: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
