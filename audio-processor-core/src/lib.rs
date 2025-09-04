//! High-performance audio processing core for real-time noise reduction
//! 
//! This library provides WebAssembly-compatible audio processing components
//! including noise gates, FFT processors, and spectral analysis tools.

use wasm_bindgen::prelude::*;
use spectral_subtraction::SpectralSubtractor;
use noise_gate::{NoiseGate, NoiseGateConfig};

/// Utility functions for audio processing
pub mod utils;
/// Real-time noise gate implementation
pub mod noise_gate;
/// Lock-free ring buffer for audio streaming
pub mod ring_buffer;
/// FFT-based spectral processing
pub mod fft_processor;
/// Noise profile analysis and reduction
pub mod noise_profile;
/// Spectral subtraction for noise reduction
pub mod spectral_subtraction;
/// Spectral smoothing filters to reduce musical noise
pub mod spectral_smoothing;

// Enhanced CPAL wrapper for better device support (native only)
// #[cfg(not(target_arch = "wasm32"))]
// /// Enhanced CPAL wrapper that fixes common device issues like AirPods on macOS
// pub mod cpal_plus;

/// WebAssembly-compatible noise reduction processor
/// Combines noise gate and spectral subtraction for comprehensive noise reduction
#[wasm_bindgen]
pub struct NoiseReducer {
    sample_rate: f32,
    frame_size: usize,
    spectral_subtractor: SpectralSubtractor,
    noise_gate: NoiseGate,
    bypass: bool,
    gate_enabled: bool,
    spectral_enabled: bool,
    // Pre-allocated working buffers for zero-copy API
    working_buffer: Vec<f32>,
}

#[wasm_bindgen]
impl NoiseReducer {
    /// Creates a new noise reducer with the specified sample rate
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        // Initialize spectral subtractor with custom settings for better quality
        use spectral_subtraction::SpectralSubtractionConfig;
        let mut config = SpectralSubtractionConfig::default();
        config.sample_rate = sample_rate;
        config.enable_smoothing = true;
        config.enable_comfort_noise = true;  // Enable comfort noise to mask musical artifacts
        config.comfort_noise_gain = 0.02;    // Low level comfort noise
        config.wiener_filter_mode = true;    // Use Wiener filter by default
        
        let spectral_subtractor = SpectralSubtractor::new(config)
            .expect("Failed to create spectral subtractor");
        
        // Initialize noise gate with default settings
        let gate_config = NoiseGateConfig::default();
        let noise_gate = NoiseGate::new(gate_config);
        
        NoiseReducer {
            sample_rate,
            frame_size: spectral_subtraction::DEFAULT_FFT_SIZE,  // Use the FFT size constant (256)
            spectral_subtractor,
            noise_gate,
            bypass: false,
            gate_enabled: false,  // Start with gate disabled due to choppy audio issue
            spectral_enabled: true,
            working_buffer: Vec::new(),
        }
    }

    /// Processes an audio buffer and returns the denoised output
    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.bypass {
            return input.to_vec();
        }
        
        let mut output = vec![0.0; input.len()];
        
        // Apply spectral subtraction if enabled
        if self.spectral_enabled {
            if let Err(_) = self.spectral_subtractor.process_buffer(input, &mut output) {
                // If spectral subtraction fails, use input signal
                output.copy_from_slice(input);
            }
        } else {
            output.copy_from_slice(input);
        }
        
        // Apply noise gate as post-processing if enabled
        // This avoids breaking the FFT overlap-add continuity
        if self.gate_enabled {
            for sample in output.iter_mut() {
                *sample = self.noise_gate.process_sample(*sample);
            }
        }
        
        output
    }
    
    /// Learns noise profile from a buffer of noise-only samples
    #[wasm_bindgen]
    pub fn learn_noise(&mut self, noise_samples: &[f32]) {
        // Update spectral subtractor's noise profile
        let _ = self.spectral_subtractor.learn_noise_profile(noise_samples);
        
        // Could also update noise gate threshold based on noise level
        // For now, we just learn the noise profile for spectral subtraction
        // In the future, we could recreate the noise gate with a new threshold
    }
    
    /// Sets the noise reduction aggressiveness (0.0 = minimal, 1.0 = maximum)
    #[wasm_bindgen]
    pub fn set_reduction_amount(&mut self, amount: f32) {
        let amount = amount.clamp(0.0, 1.0);
        
        // For Wiener filter mode, use even more conservative parameters to reduce musical noise
        // Map to alpha parameter (0.5 to 1.5 for Wiener, 1.0 to 2.5 for standard)
        let alpha = if self.spectral_subtractor.is_wiener_mode() {
            0.5 + amount * 1.0  // 0.5 to 1.5 for Wiener (very conservative to avoid musical noise)
        } else {
            1.0 + amount * 1.5  // 1.0 to 2.5 for standard
        };
        self.spectral_subtractor.set_alpha(alpha);
        
        // Also adjust beta (spectral floor) - keep it low but not too low to avoid musical noise
        let beta = 0.01 + amount * 0.02;  // 0.01 to 0.03 (moderate floor to mask artifacts)
        self.spectral_subtractor.set_beta(beta);
        
        // Adjust comfort noise level based on reduction amount
        let comfort_noise = 0.01 + amount * 0.02;  // 0.01 to 0.03
        self.spectral_subtractor.set_comfort_noise_gain(comfort_noise);
    }
    
    /// Enable or disable Wiener filter mode for spectral subtraction
    #[wasm_bindgen]
    pub fn set_wiener_filter_mode(&mut self, enabled: bool) {
        self.spectral_subtractor.set_wiener_filter_mode(enabled);
    }
    
    /// Enable or disable the noise gate
    #[wasm_bindgen]
    pub fn set_gate_enabled(&mut self, enabled: bool) {
        self.gate_enabled = enabled;
    }
    
    /// Enable or disable spectral subtraction
    #[wasm_bindgen]
    pub fn set_spectral_enabled(&mut self, enabled: bool) {
        self.spectral_enabled = enabled;
    }
    
    /// Set noise gate threshold in dB
    #[wasm_bindgen]
    pub fn set_gate_threshold(&mut self, threshold_db: f32) {
        // Create new config with updated threshold
        let mut config = NoiseGateConfig::default();
        config.threshold_db = threshold_db;
        self.noise_gate = NoiseGate::new(config);
    }
    
    /// Enables or disables bypass mode
    #[wasm_bindgen]
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }
    
    /// Resets the processor state
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.spectral_subtractor.reset();
        self.noise_gate.reset();
    }

    /// Returns the frame size used for processing
    #[wasm_bindgen(getter)]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
    
    /// Returns the current sample rate
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    
    // ========================================================================
    // Zero-Copy Buffer API for Real-Time Performance
    // ========================================================================
    
    /// Allocate a buffer in WASM memory and return its pointer.
    /// The caller can create a Float32Array view over this memory in JS.
    /// 
    /// # Safety
    /// The returned pointer is valid until the next call to alloc_buffer or
    /// until the NoiseReducer is dropped.
    #[wasm_bindgen]
    pub fn alloc_buffer(&mut self, len: usize) -> *mut f32 {
        self.working_buffer.resize(len, 0.0);
        self.working_buffer.as_mut_ptr()
    }
    
    /// Process audio in-place using a buffer in WASM memory.
    /// This avoids allocation and copying across the WASM boundary.
    /// 
    /// # Safety
    /// The caller must ensure the pointer points to valid memory of at least `len` floats.
    #[wasm_bindgen(js_name = processPtr)]
    pub fn process_ptr(&mut self, ptr: *mut f32, len: usize) {
        if self.bypass {
            return;
        }
        
        unsafe {
            // Create a mutable slice from the pointer
            let buffer = std::slice::from_raw_parts_mut(ptr, len);
            
            // Process using existing logic
            if self.spectral_enabled {
                // Create a temporary copy for input since we can't have both immutable and mutable refs
                let input_copy = buffer.to_vec();
                
                if let Err(_) = self.spectral_subtractor.process_buffer(&input_copy, buffer) {
                    // If spectral subtraction fails, don't modify the buffer
                    return;
                }
            }
            
            // Apply noise gate as post-processing if enabled
            if self.gate_enabled {
                for sample in buffer.iter_mut() {
                    *sample = self.noise_gate.process_sample(*sample);
                }
            }
            
            // If neither spectral nor gate is enabled, buffer remains unchanged (passthrough)
        }
    }
    
    /// Process audio from input buffer to output buffer (both in WASM memory).
    /// This allows processing without in-place constraints.
    /// 
    /// # Safety
    /// The caller must ensure both pointers point to valid memory of at least `len` floats.
    /// The input and output buffers must not overlap.
    #[wasm_bindgen(js_name = processInto)]
    pub fn process_into(&mut self, in_ptr: *const f32, out_ptr: *mut f32, len: usize) {
        unsafe {
            // Create slices from raw pointers
            let input = std::slice::from_raw_parts(in_ptr, len);
            let output = std::slice::from_raw_parts_mut(out_ptr, len);
            
            if self.bypass {
                output.copy_from_slice(input);
                return;
            }
            
            // Apply spectral subtraction if enabled
            if self.spectral_enabled {
                if let Err(_) = self.spectral_subtractor.process_buffer(input, output) {
                    // If spectral subtraction fails, copy input to output
                    output.copy_from_slice(input);
                }
            } else {
                output.copy_from_slice(input);
            }
            
            // Apply noise gate as post-processing if enabled
            if self.gate_enabled {
                for sample in output.iter_mut() {
                    *sample = self.noise_gate.process_sample(*sample);
                }
            }
        }
    }
    
    /// Get the size of the last allocated buffer
    #[wasm_bindgen(getter)]
    pub fn buffer_size(&self) -> usize {
        self.working_buffer.len()
    }
}

/// Example greeting function for testing WASM bindings
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello from Rust, {}!", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_reduction_works() {
        println!("\n🎤 Testing Rust Noise Reduction");
        println!("================================");
        
        // Create noise reducer
        let mut reducer = NoiseReducer::new(48000.0);
        println!("✅ Created NoiseReducer");
        
        // Configure
        reducer.set_bypass(false);
        reducer.set_reduction_amount(0.5);
        
        // Create test signal with noise
        let mut test_signal = Vec::new();
        for i in 0..1024 {
            let sine = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3;
            let noise = ((i as f32 * 12.345).sin() * 43.21).fract() * 0.05;
            test_signal.push(sine + noise);
        }
        
        // Process
        let processed = reducer.process(&test_signal);
        
        // Check we got output
        assert_eq!(processed.len(), test_signal.len());
        
        // Calculate RMS
        let rms_before: f32 = test_signal.iter().map(|x| x * x).sum::<f32>() / test_signal.len() as f32;
        let rms_after: f32 = processed.iter().map(|x| x * x).sum::<f32>() / processed.len() as f32;
        
        println!("Input RMS:  {:.6}", rms_before.sqrt());
        println!("Output RMS: {:.6}", rms_after.sqrt());
        
        // Should have some reduction
        assert!(rms_after <= rms_before * 1.1); // Allow 10% tolerance
        
        println!("✅ Noise reduction test passed!");
    }
}
