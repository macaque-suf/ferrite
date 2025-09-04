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

/// WebAssembly-compatible noise reduction processor
/// Combines noise gate and spectral subtraction for comprehensive noise reduction
#[wasm_bindgen]
pub struct NoiseReducer {
    sample_rate: f32,
    frame_size: usize,
    spectral_subtractor: SpectralSubtractor,
    noise_gate: NoiseGate,
    bypass: bool,
}

#[wasm_bindgen]
impl NoiseReducer {
    /// Creates a new noise reducer with the specified sample rate
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        // Initialize spectral subtractor with default settings
        let spectral_subtractor = SpectralSubtractor::with_sample_rate(sample_rate);
        
        // Initialize noise gate with default settings
        let gate_config = NoiseGateConfig::default();
        let noise_gate = NoiseGate::new(gate_config);
        
        NoiseReducer {
            sample_rate,
            frame_size: 512,
            spectral_subtractor,
            noise_gate,
            bypass: false,
        }
    }

    /// Processes an audio buffer and returns the denoised output
    /// Applies noise gate first, then spectral subtraction
    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.bypass {
            return input.to_vec();
        }
        
        let mut output = vec![0.0; input.len()];
        
        // Stage 1: Apply noise gate
        let mut gated = input.to_vec();
        for (i, sample) in input.iter().enumerate() {
            gated[i] = self.noise_gate.process_sample(*sample);
        }
        
        // Stage 2: Apply spectral subtraction
        if let Err(_) = self.spectral_subtractor.process_buffer(&gated, &mut output) {
            // If spectral subtraction fails, return gated signal
            return gated;
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
        
        // Map to alpha parameter (1.0 to 3.0)
        let alpha = 1.0 + amount * 2.0;
        self.spectral_subtractor.set_alpha(alpha);
        
        // For noise gate, we would need to recreate it with new parameters
        // or store the config and update it on next process
        // For now, just adjust spectral subtraction
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
}

/// Example greeting function for testing WASM bindings
#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello from Rust, {}!", name)
}
