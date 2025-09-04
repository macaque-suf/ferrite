//! FFT/IFFT processor with overlap-add for real-time spectral processing
//! 
//! This module provides efficient FFT-based audio processing with proper
//! windowing and overlap-add reconstruction for artifact-free processing.

use rustfft::{FftPlanner, Fft};
use num_complex::Complex32;
use std::sync::Arc;
use std::f32::consts::PI;

// Import the lock-free ring buffer from your ring_buffer module
use crate::ring_buffer::{RingBuffer as LockFreeRingBuffer, RingBufferError};

// ============================================================================
// Constants
// ============================================================================

/// Minimum FFT size supported
pub const MIN_FFT_SIZE: usize = 64;

/// Maximum FFT size supported  
pub const MAX_FFT_SIZE: usize = 8192;

/// Default overlap percentage for overlap-add
pub const DEFAULT_OVERLAP_PERCENT: f32 = 50.0;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum FftError {
    InvalidSize { size: usize },
    InvalidOverlap { overlap: f32 },
    BufferSizeMismatch { expected: usize, got: usize },
    NotInitialized,
}

impl std::fmt::Display for FftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FftError::InvalidSize { size } => 
                write!(f, "Invalid FFT size: {}. Must be power of 2 between {} and {}", 
                       size, MIN_FFT_SIZE, MAX_FFT_SIZE),
            FftError::InvalidOverlap { overlap } => 
                write!(f, "Invalid overlap: {}%. Must be between 0 and 95", overlap),
            FftError::BufferSizeMismatch { expected, got } => 
                write!(f, "Buffer size mismatch. Expected {}, got {}", expected, got),
            FftError::NotInitialized => 
                write!(f, "FFT processor not initialized"),
        }
    }
}

impl std::error::Error for FftError {}

// ============================================================================
// Processing Mode
// ============================================================================

/// Processing mode for the FFT processor
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingMode {
    /// Standard Overlap-Add (window only on analysis)
    OLA,
    /// Weighted Overlap-Add (root windows on both analysis and synthesis)
    WOLA,
}

// ============================================================================
// Window Function Types
// ============================================================================

/// Supported window functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hann (Hanning) window - good for general purpose
    Hann,
    /// Hamming window - slightly better frequency resolution
    Hamming,
    /// Blackman window - better stopband attenuation
    Blackman,
    /// Rectangular (no window) - maximum frequency resolution but spectral leakage
    Rectangular,
    /// Tukey window with configurable taper (0.0 = rectangular, 1.0 = Hann)
    Tukey(f32),
}

impl WindowType {
    /// Generate window coefficients for the given size
    pub fn generate(&self, size: usize) -> Vec<f32> {
        let mut window = vec![0.0; size];
        self.generate_into(&mut window);
        window
    }
    
    /// Generate window coefficients into existing buffer
    pub fn generate_into(&self, window: &mut [f32]) {
        match self {
            WindowType::Hann => hann_window_into(window),
            WindowType::Hamming => hamming_window_into(window),
            WindowType::Blackman => blackman_window_into(window),
            WindowType::Rectangular => window.fill(1.0),
            WindowType::Tukey(alpha) => tukey_window_into(window, *alpha),
        }
    }
    
    /// Generate square root of window (for WOLA mode)
    pub fn generate_root(&self, size: usize) -> Vec<f32> {
        self.generate(size).iter().map(|&w| w.sqrt()).collect()
    }
    
    /// Get the recommended overlap percentage for this window
    pub fn recommended_overlap(&self) -> f32 {
        match self {
            WindowType::Hann => 50.0,
            WindowType::Hamming => 50.0,
            WindowType::Blackman => 66.7,
            WindowType::Rectangular => 0.0,
            WindowType::Tukey(alpha) => 50.0 * alpha,
        }
    }
}

// ============================================================================
// Window Functions Implementation
// ============================================================================

/// Generate Hann window coefficients
fn hann_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    let scale = std::f32::consts::TAU / (size - 1) as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let t = i as f32 * scale;
        *w = 0.5 * (1.0 - t.cos());
    }
}

/// Generate Hamming window coefficients
fn hamming_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    let scale = std::f32::consts::TAU / (size - 1) as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let t = i as f32 * scale;
        *w = 0.54 - 0.46 * t.cos();
    }
}

/// Generate Blackman window coefficients
fn blackman_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    let scale = std::f32::consts::TAU / (size - 1) as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let t = i as f32 * scale;
        *w = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
    }
}

/// Generate Tukey window coefficients
fn tukey_window_into(window: &mut [f32], alpha: f32) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    let alpha = alpha.clamp(0.0, 1.0);
    
    if alpha == 0.0 {
        window.fill(1.0);
        return;
    }
    
    let taper_length = (alpha * (size - 1) as f32 / 2.0).round() as usize;
    
    for (i, w) in window.iter_mut().enumerate() {
        *w = if i < taper_length {
            let x = i as f32 / taper_length as f32;
            0.5 * (1.0 - (PI * x).cos())
        } else if i >= size - taper_length {
            let x = (size - 1 - i) as f32 / taper_length as f32;
            0.5 * (1.0 - (PI * x).cos())
        } else {
            1.0
        };
    }
}

/// Apply window to signal
fn apply_window(signal: &mut [f32], window: &[f32]) -> Result<(), FftError> {
    if signal.len() != window.len() {
        return Err(FftError::BufferSizeMismatch {
            expected: window.len(),
            got: signal.len(),
        });
    }
    
    for (s, w) in signal.iter_mut().zip(window.iter()) {
        *s *= w;
    }
    
    Ok(())
}

// ============================================================================
// FFT Processor
// ============================================================================

/// FFT processor with configurable size and window function
pub struct FftProcessor {
    // FFT configuration
    fft_size: usize,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
    
    // Processing mode
    mode: ProcessingMode,
    
    // Windows
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    window_type: WindowType,
    
    // Overlap-add parameters
    hop_size: usize,
    overlap_percent: f32,
    
    // Working buffers (pre-allocated)
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    complex_buffer: Vec<Complex32>,
    spectrum_buffer: Vec<Complex32>,
    
    // State
    initialized: bool,
}

impl FftProcessor {
    /// Create a new FFT processor with specified size
    pub fn new(fft_size: usize) -> Result<Self, FftError> {
        Self::with_window(fft_size, WindowType::Hann, DEFAULT_OVERLAP_PERCENT, ProcessingMode::OLA)
    }
    
    /// Create FFT processor with custom window and overlap
    pub fn with_window(
        fft_size: usize,
        window_type: WindowType,
        overlap_percent: f32,
        mode: ProcessingMode,
    ) -> Result<Self, FftError> {
        // Validate FFT size
        if fft_size < MIN_FFT_SIZE || fft_size > MAX_FFT_SIZE {
            return Err(FftError::InvalidSize { size: fft_size });
        }
        
        if !fft_size.is_power_of_two() {
            return Err(FftError::InvalidSize { size: fft_size });
        }
        
        // Validate overlap
        if !(0.0..=95.0).contains(&overlap_percent) {
            return Err(FftError::InvalidOverlap { overlap: overlap_percent });
        }
        
        // Create FFT planners
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let fft_inverse = planner.plan_fft_inverse(fft_size);
        
        // Generate windows based on mode
        let (analysis_window, synthesis_window) = match mode {
            ProcessingMode::OLA => {
                // Standard OLA: window only on analysis
                let window = window_type.generate(fft_size);
                let unity_window = vec![1.0; fft_size];
                (window, unity_window)
            }
            ProcessingMode::WOLA => {
                // WOLA: use root windows on both
                let root_window = window_type.generate_root(fft_size);
                (root_window.clone(), root_window)
            }
        };
        
        // Calculate hop size
        let hop_size = ((fft_size as f32) * (1.0 - overlap_percent / 100.0)) as usize;
        let hop_size = hop_size.max(1);
        
        Ok(Self {
            fft_size,
            fft_forward,
            fft_inverse,
            mode,
            analysis_window,
            synthesis_window,
            window_type,
            hop_size,
            overlap_percent,
            input_buffer: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size],
            complex_buffer: vec![Complex32::new(0.0, 0.0); fft_size],
            spectrum_buffer: vec![Complex32::new(0.0, 0.0); fft_size],
            initialized: true,
        })
    }
    
    /// Get the FFT size
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
    
    /// Get the hop size (samples between successive FFT frames)
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
    
    /// Get the current overlap percentage
    #[inline]
    pub fn overlap_percent(&self) -> f32 {
        self.overlap_percent
    }
    
    /// Reset all buffers
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.complex_buffer.fill(Complex32::new(0.0, 0.0));
        self.spectrum_buffer.fill(Complex32::new(0.0, 0.0));
    }
    
    /// Process a single FFT frame (forward transform)
    pub fn forward(&mut self, input: &[f32]) -> Result<&[Complex32], FftError> {
        if !self.initialized {
            return Err(FftError::NotInitialized);
        }
        
        if input.len() != self.fft_size {
            return Err(FftError::BufferSizeMismatch {
                expected: self.fft_size,
                got: input.len(),
            });
        }
        
        // Copy and window the input
        self.input_buffer.copy_from_slice(input);
        apply_window(&mut self.input_buffer, &self.analysis_window)?;
        
        // Convert to complex
        for (i, &sample) in self.input_buffer.iter().enumerate() {
            self.complex_buffer[i] = Complex32::new(sample, 0.0);
        }
        
        // Perform FFT
        self.fft_forward.process(&mut self.complex_buffer);
        
        // Apply scaling (1/sqrt(N) for both forward and inverse)
        let scale = 1.0 / (self.fft_size as f32).sqrt();
        for (i, &val) in self.complex_buffer.iter().enumerate() {
            self.spectrum_buffer[i] = val * scale;
        }
        
        Ok(&self.spectrum_buffer)
    }
    
    /// Process inverse FFT (inverse transform)
    pub fn inverse(&mut self, spectrum: &[Complex32]) -> Result<&[f32], FftError> {
        if !self.initialized {
            return Err(FftError::NotInitialized);
        }
        
        if spectrum.len() != self.fft_size {
            return Err(FftError::BufferSizeMismatch {
                expected: self.fft_size,
                got: spectrum.len(),
            });
        }
        
        // Copy spectrum to working buffer
        self.complex_buffer.copy_from_slice(spectrum);
        
        // Perform inverse FFT
        self.fft_inverse.process(&mut self.complex_buffer);
        
        // Convert back to real and apply scaling
        let scale = 1.0 / (self.fft_size as f32).sqrt();
        for (i, val) in self.complex_buffer.iter().enumerate() {
            self.output_buffer[i] = val.re * scale;
        }
        
        // Apply synthesis window only in WOLA mode
        if self.mode == ProcessingMode::WOLA {
            apply_window(&mut self.output_buffer, &self.synthesis_window)?;
        }
        
        Ok(&self.output_buffer)
    }
    
    /// Get magnitude spectrum from complex spectrum
    pub fn get_magnitude_spectrum(spectrum: &[Complex32]) -> Vec<f32> {
        spectrum.iter().map(|c| c.norm()).collect()
    }
    
    /// Get phase spectrum from complex spectrum
    pub fn get_phase_spectrum(spectrum: &[Complex32]) -> Vec<f32> {
        spectrum.iter().map(|c| c.arg()).collect()
    }
    
    /// Convert magnitude and phase back to complex spectrum
    pub fn from_magnitude_phase(magnitudes: &[f32], phases: &[f32]) -> Vec<Complex32> {
        magnitudes.iter()
            .zip(phases.iter())
            .map(|(&mag, &phase)| Complex32::from_polar(mag, phase))
            .collect()
    }
    
    /// Get frequency bin for a given index
    #[inline]
    pub fn bin_frequency(&self, bin: usize, sample_rate: f32) -> f32 {
        (bin as f32 * sample_rate) / (self.fft_size as f32)
    }
    
    /// Get all frequency bins
    pub fn frequency_bins(&self, sample_rate: f32) -> Vec<f32> {
        (0..self.fft_size/2 + 1)
            .map(|i| self.bin_frequency(i, sample_rate))
            .collect()
    }
    
    /// Calculate window compensation factor for unity gain in overlap-add
    pub fn calculate_window_compensation(&self) -> f32 {
        // For OLA mode, calculate the COLA sum at steady state
        // For WOLA mode, this includes both analysis and synthesis windows
        let cola_gain = calculate_cola_gain(&self.analysis_window, self.overlap_percent);
        
        // Additional compensation for WOLA mode where synthesis window is also applied
        let synthesis_compensation = if self.mode == ProcessingMode::WOLA {
            // In WOLA, both windows are sqrt windows, so their product gives the original window
            // The COLA gain already accounts for the analysis window
            1.0
        } else {
            // In OLA mode, only analysis window affects the gain
            1.0
        };
        
        // Return compensation factor
        // If COLA gain is too small (near zero), avoid division by zero
        if cola_gain.abs() < 0.01 {
            1.0  // No compensation if COLA gain is too small
        } else {
            synthesis_compensation / cola_gain
        }
    }
}

// ============================================================================
// Overlap-Add Processor (FIXED VERSION)
// ============================================================================

/// Overlap-add processor for streaming FFT processing with proper sliding window
pub struct OverlapAddProcessor {
    fft_processor: FftProcessor,
    
    // Ring buffers for streaming (using lock-free implementation)
    input_buffer: LockFreeRingBuffer,
    output_buffer: LockFreeRingBuffer,
    
    // Sliding window for FFT input
    fft_input_window: Vec<f32>,
    
    // Overlap buffer for output accumulation
    overlap_buffer: Vec<f32>,
    
    // Reusable buffer for reading hop_size samples (performance optimization)
    hop_buffer: Vec<f32>,
    
    // Work buffers
    spectrum_work: Vec<Complex32>,
    
    // Processing callback
    process_callback: Option<Box<dyn FnMut(&mut [Complex32])>>,
    
    // Window compensation factor for unity gain
    window_compensation: f32,
    
    // State tracking
    frames_processed: usize,
}

impl OverlapAddProcessor {
    /// Create a new overlap-add processor
    pub fn new(
        fft_size: usize,
        window_type: WindowType,
        overlap_percent: f32,
        mode: ProcessingMode,
    ) -> Result<Self, FftError> {
        let fft_processor = FftProcessor::with_window(fft_size, window_type, overlap_percent, mode)?;
        
        // Calculate window compensation factor for unity gain
        let window_compensation = fft_processor.calculate_window_compensation();
        let hop_size = fft_processor.hop_size();
        
        // Create ring buffers with sufficient capacity
        // Note: Consider using StaticRingBuffer<SIZE> for compile-time optimizations
        // if buffer size is known at compile time
        let buffer_size = fft_size * 8;
        let input_buffer = LockFreeRingBuffer::new(buffer_size)
            .map_err(|_| FftError::InvalidSize { size: buffer_size })?;
        let output_buffer = LockFreeRingBuffer::new(buffer_size)
            .map_err(|_| FftError::InvalidSize { size: buffer_size })?;
        
        Ok(Self {
            fft_processor,
            input_buffer,
            output_buffer,
            fft_input_window: vec![0.0; fft_size],
            overlap_buffer: vec![0.0; fft_size],
            hop_buffer: vec![0.0; hop_size],  // Pre-allocated for performance
            spectrum_work: vec![Complex32::new(0.0, 0.0); fft_size],
            process_callback: None,
            window_compensation,
            frames_processed: 0,
        })
    }
    
    /// Set the spectral processing callback
    pub fn set_process_callback<F>(&mut self, callback: F)
    where
        F: FnMut(&mut [Complex32]) + 'static,
    {
        self.process_callback = Some(Box::new(callback));
    }
    
    /// Process audio samples with overlap-add
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, FftError> {
        // Add input to buffer
        self.input_buffer.write_available(input);
        
        let fft_size = self.fft_processor.fft_size;
        let hop_size = self.fft_processor.hop_size;
        
        output.fill(0.0);
        let mut output_written = 0;
        
        // Process complete frames using sliding window
        while self.input_buffer.available() >= hop_size {
            // Special handling for first frame
            if self.frames_processed == 0 && self.input_buffer.available() >= fft_size {
                // Read full FFT frame for the first time
                if self.input_buffer.read(&mut self.fft_input_window).is_err() {
                    break;
                }
            } else if self.frames_processed > 0 {
                // Sliding window: shift left by hop_size
                self.fft_input_window.rotate_left(hop_size);
                
                // Read hop_size new samples into reusable buffer (no allocation)
                if self.input_buffer.read(&mut self.hop_buffer).is_err() {
                    break;
                }
                
                // Place new samples at the end
                self.fft_input_window[fft_size - hop_size..].copy_from_slice(&self.hop_buffer);
            } else {
                // Not enough samples for first frame yet
                break;
            }
            
            // Forward FFT (includes windowing)
            let spectrum = self.fft_processor.forward(&self.fft_input_window)?;
            self.spectrum_work.copy_from_slice(spectrum);
            
            // Apply spectral processing if callback is set
            if let Some(ref mut callback) = self.process_callback {
                callback(&mut self.spectrum_work);
            }
            
            // Inverse FFT
            let frame_result = self.fft_processor.inverse(&self.spectrum_work)?;
            
            // Overlap-add into accumulator with window compensation
            for i in 0..fft_size {
                self.overlap_buffer[i] += frame_result[i] * self.window_compensation;
            }
            
            // Output the first hop_size samples
            let samples_to_output = hop_size.min(output.len() - output_written);
            for i in 0..samples_to_output {
                output[output_written + i] = self.overlap_buffer[i];
            }
            output_written += samples_to_output;
            
            // Shift the overlap buffer left by hop_size and zero the end
            self.overlap_buffer.rotate_left(hop_size);
            self.overlap_buffer[fft_size - hop_size..].fill(0.0);
            
            self.frames_processed += 1;
        }
        
        Ok(output_written)
    }
    
    /// Get the latency in samples
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.fft_processor.fft_size
    }
    
    /// Reset all buffers
    pub fn reset(&mut self) {
        self.fft_processor.reset();
        
        // Note: We recreate ring buffers here because the current RingBuffer implementation
        // only provides reset(&mut self), which we can't call on a non-mutable reference.
        // In a production system, you'd want to modify RingBuffer to provide a thread-safe
        // reset method that works with &self (using interior mutability).
        let capacity = self.input_buffer.capacity();
        self.input_buffer = LockFreeRingBuffer::new(capacity).unwrap();
        self.output_buffer = LockFreeRingBuffer::new(capacity).unwrap();
        
        self.fft_input_window.fill(0.0);
        self.overlap_buffer.fill(0.0);
        self.hop_buffer.fill(0.0);
        self.spectrum_work.fill(Complex32::new(0.0, 0.0));
        self.frames_processed = 0;
    }
}

// ============================================================================
// Phase Vocoder for Pitch Shifting
// ============================================================================

/// Phase vocoder for pitch shifting and time stretching
pub struct PhaseVocoder {
    fft_size: usize,
    hop_size: usize,
    sample_rate: f32,
    
    // Phase tracking
    last_phase_input: Vec<f32>,
    last_phase_output: Vec<f32>,
    phase_accumulator: Vec<f32>,
    
    // Frequency analysis
    bin_frequencies: Vec<f32>,
    expected_phase_increment: Vec<f32>,
}

impl PhaseVocoder {
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: f32) -> Self {
        let num_bins = fft_size / 2 + 1;
        
        // Pre-calculate bin frequencies and expected phase increments
        let mut bin_frequencies = vec![0.0; num_bins];
        let mut expected_phase_increment = vec![0.0; num_bins];
        
        for bin in 0..num_bins {
            bin_frequencies[bin] = (bin as f32 * sample_rate) / fft_size as f32;
            expected_phase_increment[bin] = 2.0 * PI * (bin as f32) * (hop_size as f32) / (fft_size as f32);
        }
        
        Self {
            fft_size,
            hop_size,
            sample_rate,
            last_phase_input: vec![0.0; num_bins],
            last_phase_output: vec![0.0; num_bins],
            phase_accumulator: vec![0.0; num_bins],
            bin_frequencies,
            expected_phase_increment,
        }
    }
    
    /// Process spectrum for pitch shifting
    pub fn process_pitch_shift(&mut self, spectrum: &mut [Complex32], pitch_ratio: f32) {
        let num_bins = self.fft_size / 2 + 1;
        
        // Extract magnitude and phase
        let magnitudes: Vec<f32> = spectrum[..num_bins].iter().map(|c| c.norm()).collect();
        let phases: Vec<f32> = spectrum[..num_bins].iter().map(|c| c.arg()).collect();
        
        // Create output spectrum
        let mut output_magnitudes = vec![0.0; num_bins];
        let mut output_phases = vec![0.0; num_bins];
        
        // Process each bin
        for bin in 0..num_bins {
            // Calculate phase difference
            let mut phase_diff = phases[bin] - self.last_phase_input[bin];
            self.last_phase_input[bin] = phases[bin];
            
            // Unwrap phase difference
            phase_diff -= self.expected_phase_increment[bin];
            
            // Wrap to [-pi, pi]
            let qpd = (phase_diff / PI) as i32;
            if qpd >= 0 {
                phase_diff -= PI * 2.0 * ((qpd + 1) / 2) as f32;
            } else {
                phase_diff -= PI * 2.0 * ((qpd - 1) / 2) as f32;
            }
            
            // Calculate true frequency deviation
            let freq_deviation = phase_diff * (self.fft_size as f32) / (2.0 * PI * self.hop_size as f32);
            let true_freq = self.bin_frequencies[bin] + freq_deviation;
            
            // Shift frequency for pitch change
            let shifted_freq = true_freq * pitch_ratio;
            let target_bin = ((shifted_freq * self.fft_size as f32) / self.sample_rate) as usize;
            
            if target_bin < num_bins {
                output_magnitudes[target_bin] += magnitudes[bin];
                
                // Calculate output phase
                let phase_increment = 2.0 * PI * shifted_freq * (self.hop_size as f32) / self.sample_rate;
                self.phase_accumulator[target_bin] += phase_increment;
                output_phases[target_bin] = self.phase_accumulator[target_bin];
            }
        }
        
        // Convert back to complex spectrum
        for bin in 0..num_bins {
            spectrum[bin] = Complex32::from_polar(output_magnitudes[bin], output_phases[bin]);
            
            // Mirror for negative frequencies (conjugate symmetry)
            if bin > 0 && bin < num_bins - 1 {
                let mirror_bin = self.fft_size - bin;
                spectrum[mirror_bin] = spectrum[bin].conj();
            }
        }
    }
    
    /// Reset phase tracking
    pub fn reset(&mut self) {
        self.last_phase_input.fill(0.0);
        self.last_phase_output.fill(0.0);
        self.phase_accumulator.fill(0.0);
    }
    
    /// Process pitch shift while preserving formants (spectral envelope)
    /// This is important for natural-sounding voice pitch shifting
    pub fn process_pitch_shift_preserve_formants(
        &mut self,
        spectrum: &mut [Complex32],
        pitch_ratio: f32,
    ) {
        let num_bins = self.fft_size / 2 + 1;
        
        // Extract spectral envelope (formants)
        let magnitudes: Vec<f32> = spectrum[..num_bins].iter().map(|c| c.norm()).collect();
        
        // Extract spectral envelope with adaptive smoothing
        let envelope = self.extract_spectral_envelope(&magnitudes);
        
        // Apply pitch shift
        self.process_pitch_shift(spectrum, pitch_ratio);
        
        // Reapply original spectral envelope
        // This preserves formants while shifting pitch
        let shifted_magnitudes: Vec<f32> = spectrum[..num_bins].iter().map(|c| c.norm()).collect();
        for i in 0..num_bins {
            if shifted_magnitudes[i] > 0.001 && envelope[i] > 0.001 {
                let correction = envelope[i] / shifted_magnitudes[i];
                spectrum[i] *= correction.min(10.0); // Limit correction to avoid instability
                
                // Maintain conjugate symmetry
                if i > 0 && i < num_bins - 1 {
                    spectrum[self.fft_size - i] = spectrum[i].conj();
                }
            }
        }
    }
    
    /// Extract spectral envelope using logarithmic domain smoothing
    fn extract_spectral_envelope(&self, magnitudes: &[f32]) -> Vec<f32> {
        let num_bins = magnitudes.len();
        
        // Adaptive smoothing width based on FFT size
        // Typically 5-10% of spectrum width
        let smooth_width = ((num_bins as f32 * 0.07) as usize).max(3);
        
        // Convert to log domain for better envelope extraction
        let log_mags: Vec<f32> = magnitudes.iter()
            .map(|&m| (m.max(1e-10)).ln())
            .collect();
        
        // Apply smoothing in log domain
        let mut envelope = vec![0.0; num_bins];
        for i in 0..num_bins {
            let start = i.saturating_sub(smooth_width);
            let end = (i + smooth_width + 1).min(num_bins);
            
            // Weighted average with triangular window
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for j in start..end {
                let distance = ((j as i32) - (i as i32)).abs() as f32;
                let weight = (1.0 - distance / (smooth_width as f32)).max(0.0);
                sum += log_mags[j] * weight;
                weight_sum += weight;
            }
            
            if weight_sum > 0.0 {
                // Convert back from log domain
                envelope[i] = (sum / weight_sum).exp();
            } else {
                envelope[i] = magnitudes[i];
            }
        }
        
        envelope
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if a number is a power of two
#[inline]
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Calculate proper window gain for COLA
/// 
/// This checks that the sum of overlapped windows equals a constant (amplitude COLA).
/// Note: For power preservation, you would check sum of squared windows, but for
/// standard overlap-add we need amplitude COLA, not power COLA.
fn calculate_cola_gain(window: &[f32], overlap_percent: f32) -> f32 {
    let size = window.len();
    let hop_size = ((size as f32) * (1.0 - overlap_percent / 100.0)) as usize;
    let hop_size = hop_size.max(1);
    
    // Check COLA compliance at all points in hop interval
    let mut max_sum = 0.0f32;
    let mut min_sum = f32::MAX;
    
    for offset in 0..hop_size {
        let mut sum = 0.0;
        let mut frame_idx = 0;
        
        // Sum all overlapping windows at this offset
        loop {
            let sample_pos = frame_idx * hop_size + offset;
            if sample_pos >= size {
                break;
            }
            sum += window[sample_pos];
            frame_idx += 1;
        }
        
        max_sum = max_sum.max(sum);
        min_sum = min_sum.min(sum);
    }
    
    // Return average for normalization
    (max_sum + min_sum) / 2.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    
    const EPSILON: f32 = 1e-5;
    
    #[test]
    fn test_window_types() {
        let size = 256;
        
        let windows = vec![
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::Rectangular,
            WindowType::Tukey(0.5),
        ];
        
        for window_type in windows {
            let window = window_type.generate(size);
            assert_eq!(window.len(), size);
            
            match window_type {
                WindowType::Rectangular => {
                    assert!(window.iter().all(|&w| (w - 1.0).abs() < EPSILON));
                }
                WindowType::Hann | WindowType::Hamming | WindowType::Blackman => {
                    assert!(window[0] < 0.1);
                    assert!(window[size - 1] < 0.1);
                    assert!(window[size / 2] > 0.9);
                }
                WindowType::Tukey(_) => {
                    assert!((window[size / 2] - 1.0).abs() < EPSILON);
                }
            }
        }
    }
    
    #[test]
    fn test_fft_processor_creation() {
        // Valid sizes
        assert!(FftProcessor::new(256).is_ok());
        assert!(FftProcessor::new(512).is_ok());
        assert!(FftProcessor::new(1024).is_ok());
        
        // Invalid sizes
        assert!(FftProcessor::new(0).is_err());
        assert!(FftProcessor::new(63).is_err());
        assert!(FftProcessor::new(257).is_err());
        assert!(FftProcessor::new(16384).is_err());
        
        // Test with different windows
        assert!(FftProcessor::with_window(512, WindowType::Hann, 50.0, ProcessingMode::OLA).is_ok());
        assert!(FftProcessor::with_window(512, WindowType::Blackman, 75.0, ProcessingMode::WOLA).is_ok());
    }
    
    #[test]
    fn test_forward_inverse_perfect_reconstruction() {
        let mut processor = FftProcessor::with_window(
            512,
            WindowType::Rectangular,
            0.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        // Generate test signal
        let mut input = vec![0.0; 512];
        for i in 0..512 {
            input[i] = (2.0 * PI * 10.0 * i as f32 / 512.0).sin();
        }
        
        // Forward FFT
        let spectrum = processor.forward(&input).unwrap().to_vec();
        
        // Inverse FFT
        let output = processor.inverse(&spectrum).unwrap();
        
        // Check reconstruction
        for i in 0..512 {
            assert!((input[i] - output[i]).abs() < 0.001);
        }
    }
    
    #[test]
    fn test_magnitude_phase_conversion() {
        let mut processor = FftProcessor::new(256).unwrap();
        
        let input: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * 5.0 * i as f32 / 256.0).sin())
            .collect();
        
        let spectrum = processor.forward(&input).unwrap();
        
        let magnitudes = FftProcessor::get_magnitude_spectrum(spectrum);
        let phases = FftProcessor::get_phase_spectrum(spectrum);
        
        let reconstructed = FftProcessor::from_magnitude_phase(&magnitudes, &phases);
        
        for i in 0..spectrum.len() {
            assert!((spectrum[i].re - reconstructed[i].re).abs() < EPSILON);
            assert!((spectrum[i].im - reconstructed[i].im).abs() < EPSILON);
        }
    }
    
    #[test]
    fn test_hop_size_calculation() {
        let test_cases = vec![
            (512, 50.0, 256),
            (512, 75.0, 128),
            (512, 0.0, 512),
            (1024, 50.0, 512),
        ];
        
        for (fft_size, overlap, expected_hop) in test_cases {
            let processor = FftProcessor::with_window(
                fft_size,
                WindowType::Hann,
                overlap,
                ProcessingMode::OLA,
            ).unwrap();
            
            assert_eq!(processor.hop_size(), expected_hop);
        }
    }
    
    #[test]
    fn test_cola_compliance() {
        // Test Hann window with 50% overlap
        let window = WindowType::Hann.generate(512);
        let gain = calculate_cola_gain(&window, 50.0);
        
        // For Hann with 50% overlap, gain should be close to 1.0
        assert!((gain - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_overlap_add_processor() {
        let mut processor = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        println!("\n=== Overlap-Add Processor Test Debug ===");
        println!("FFT size: {}", processor.fft_processor.fft_size);
        println!("Hop size: {}", processor.fft_processor.hop_size);
        println!("Overlap: {}%", processor.fft_processor.overlap_percent);
        
        // Identity processing
        processor.set_process_callback(|_spectrum| {});
        
        // Generate test signal
        let input: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();
        
        println!("Input signal length: {}", input.len());
        println!("First 10 input samples: {:?}", &input[..10]);
        
        let mut output = vec![0.0; 2048];
        
        // Process in chunks
        let chunk_size = 256;
        let mut out_pos = 0;
        let mut total_samples_processed = 0;
        
        for (chunk_idx, i) in (0..input.len()).step_by(chunk_size).enumerate() {
            let end = (i + chunk_size).min(input.len());
            let chunk = &input[i..end];
            let mut temp_out = vec![0.0; chunk_size];
            let processed = processor.process(chunk, &mut temp_out).unwrap();
            
            println!("Chunk {}: input[{}..{}] (len={}), processed {} output samples", 
                     chunk_idx, i, end, chunk.len(), processed);
            
            if processed > 0 {
                println!("  First non-zero output in chunk: {:?}", 
                         temp_out[..processed.min(5)].to_vec());
            }
            
            for j in 0..processed {
                if out_pos + j < output.len() {
                    output[out_pos + j] = temp_out[j];
                }
            }
            out_pos += processed;
            total_samples_processed += processed;
        }
        
        println!("\nTotal samples processed: {}", total_samples_processed);
        println!("Output position reached: {}", out_pos);
        
        // Check energy preservation in steady state
        // Skip the initial latency period and the end
        let latency = processor.latency_samples();
        if out_pos > latency * 2 {
            let check_start = latency;
            let check_end = (out_pos - latency).min(input.len() - latency);
            
            let input_energy: f32 = input[check_start..check_end]
                .iter().map(|x| x * x).sum();
            let output_energy: f32 = output[check_start..check_end]
                .iter().map(|x| x * x).sum();
            
            println!("\n=== Energy Analysis ===");
            println!("Input energy (steady state): {}", input_energy);
            println!("Output energy (steady state): {}", output_energy);
            
            if input_energy > 0.0 {
                let energy_ratio = (input_energy - output_energy).abs() / input_energy;
                println!("Energy ratio: {:.4}", energy_ratio);
                println!("Threshold: 0.15");
                println!("Pass: {}", energy_ratio < 0.15);
                
                assert!(energy_ratio < 0.15, 
                        "Energy ratio {} exceeds threshold 0.15", energy_ratio);
            }
        }
    }
    
    #[test]
    fn test_phase_vocoder_pitch_shift() {
        let mut vocoder = PhaseVocoder::new(512, 256, 48000.0);
        
        // Create a simple spectrum
        let mut spectrum = vec![Complex32::new(0.0, 0.0); 512];
        
        // Add a single frequency component
        let bin = 10;
        spectrum[bin] = Complex32::from_polar(1.0, 0.0);
        spectrum[512 - bin] = spectrum[bin].conj(); // Conjugate symmetry
        
        // Pitch shift up by 2x
        vocoder.process_pitch_shift(&mut spectrum, 2.0);
        
        // The energy should have moved to approximately bin 20
        let magnitudes = FftProcessor::get_magnitude_spectrum(&spectrum);
        
        // Find peak
        let mut max_bin = 0;
        let mut max_mag = 0.0;
        for (i, &mag) in magnitudes.iter().enumerate().take(256) {
            if mag > max_mag {
                max_mag = mag;
                max_bin = i;
            }
        }
        
        // Should be approximately at 2x the original frequency
        assert!((max_bin as i32 - 20).abs() <= 1);
    }
    
    #[test]
    fn test_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(64));
        assert!(is_power_of_two(512));
        
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(100));
    }
}
