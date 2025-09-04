use wasm_bindgen::prelude::*;

pub mod utils;
pub mod noise_gate;
pub mod ring_buffer;
pub mod fft_processor;
pub mod noise_profile;

#[wasm_bindgen]
pub struct NoiseReducer {
    sample_rate: f32,
    frame_size: usize,
}

#[wasm_bindgen]
impl NoiseReducer {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        NoiseReducer {
            sample_rate,
            frame_size: 256,
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        // P0: Simple spectral gating as proof of concept
        // Use the sample_rate to avoid warning
        let _ = self.sample_rate;  // Add this to suppress warning
        input.to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello from Rust, {}!", name)
}
