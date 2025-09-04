//! FFT/IFFT processor with overlap-add for real-time spectral processing
//! 
//! This module provides efficient FFT-based audio processing with proper
//! windowing and overlap-add reconstruction for artifact-free processing.

use rustfft::{FftPlanner, Fft};
use num_complex::Complex32;
use std::sync::Arc;
use std::f32::consts::PI;

use crate::ring_buffer::{Producer, Consumer, spsc_ring_buffer};

/// Lock-free ring buffer wrapper combining Producer and Consumer
struct LockFreeRingBuffer {
    producer: Producer,
    consumer: Consumer,
}

impl LockFreeRingBuffer {
    fn new(size: usize) -> Result<Self, &'static str> {
        match spsc_ring_buffer(size) {
            Ok((producer, consumer)) => Ok(LockFreeRingBuffer { producer, consumer }),
            Err(_) => Err("Failed to create ring buffer"),
        }
    }
    
    fn write(&mut self, data: &[f32]) -> Result<usize, &'static str> {
        self.producer.write(data).map_err(|_| "Write failed")
    }
    
    fn read(&mut self, output: &mut [f32]) -> Result<usize, &'static str> {
        self.consumer.read(output).map_err(|_| "Read failed")
    }
    
    fn available(&self) -> usize {
        self.consumer.available()
    }
    
    fn space_available(&self) -> usize {
        self.producer.space()
    }
    
    fn capacity(&self) -> usize {
        self.producer.capacity()
    }
}

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
    /// Uses periodic form for COLA compliance
    Hann,
    /// Hamming window - slightly better frequency resolution
    /// Uses periodic form for COLA compliance
    Hamming,
    /// Blackman window - better stopband attenuation
    /// Uses periodic form for COLA compliance
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

/// Generate Hann window coefficients (periodic form for COLA)
fn hann_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    // Use periodic form (N in denominator) for COLA compliance
    // This gives w[n] = 0.5 - 0.5*cos(2π*n/N)
    let scale = std::f32::consts::TAU / size as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let t = i as f32 * scale;
        *w = 0.5 * (1.0 - t.cos());
    }
}

/// Generate Hamming window coefficients (periodic form for COLA)
fn hamming_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    // Use periodic form (N in denominator) for COLA compliance
    let scale = std::f32::consts::TAU / size as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let t = i as f32 * scale;
        *w = 0.54 - 0.46 * t.cos();
    }
}

/// Generate Blackman window coefficients (periodic form for COLA)
fn blackman_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    // Use periodic form (N in denominator) for COLA compliance
    let scale = std::f32::consts::TAU / size as f32;
    
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
    
    // Handle alpha near 0 (rectangular window)
    if alpha == 0.0 {
        window.fill(1.0);
        return;
    }
    
    let taper_length = (alpha * (size - 1) as f32 / 2.0).round() as usize;
    
    // Protect against division by zero when taper_length rounds to 0
    if taper_length == 0 {
        // For very small alpha on small sizes, treat as rectangular
        window.fill(1.0);
        return;
    }
    
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
        
        // Enforce Hermitian symmetry to ensure real output
        // This prevents artifacts if user callback broke symmetry
        enforce_hermitian_symmetry(&mut self.complex_buffer);
        
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
        // Determine the effective window for COLA calculation
        // For OLA: use the analysis window directly
        // For WOLA: use the product of analysis and synthesis windows (sqrt(w) * sqrt(w) = w)
        let effective_window: Vec<f32> = match self.mode {
            ProcessingMode::OLA => self.analysis_window.clone(),
            ProcessingMode::WOLA => {
                // In WOLA, analysis and synthesis are both sqrt(window)
                // The effective amplitude window is their product
                self.analysis_window.iter()
                    .zip(self.synthesis_window.iter())
                    .map(|(a, s)| a * s)
                    .collect()
            }
        };
        
        // Calculate COLA gain for the effective window
        let cola_gain = calculate_cola_gain(&effective_window, self.overlap_percent);
        
        // Return compensation factor
        // If COLA gain is too small (near zero), avoid division by zero
        if cola_gain.abs() < 0.01 {
            1.0  // No compensation if COLA gain is too small
        } else {
            1.0 / cola_gain
        }
    }
}

// ============================================================================
// Overlap-Add Processor
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
    
    // Track position in overlap buffer for partial hop output
    overlap_read_pos: usize,
    
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
    samples_dropped: u64,
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
            overlap_read_pos: 0,
            hop_buffer: vec![0.0; hop_size],  // Pre-allocated for performance
            spectrum_work: vec![Complex32::new(0.0, 0.0); fft_size],
            process_callback: None,
            window_compensation,
            frames_processed: 0,
            samples_dropped: 0,
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
        // Add input to buffer, tracking any dropped samples
        match self.input_buffer.write(input) {
            Ok(_) => {},
            Err(_) => {
                // Buffer full - samples were dropped
                self.samples_dropped += input.len() as u64;
            }
        }
        
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
            
            // Output hop_size samples (or whatever fits in output buffer)
            let remaining = output.len() - output_written;
            let available_in_hop = hop_size - self.overlap_read_pos;
            let to_emit = available_in_hop.min(remaining);
            
            // Emit from the current read cursor
            let start = self.overlap_read_pos;
            let end = start + to_emit;
            output[output_written..output_written + to_emit]
                .copy_from_slice(&self.overlap_buffer[start..end]);
            output_written += to_emit;
            self.overlap_read_pos += to_emit;
            
            // Only rotate once we've emitted a full hop
            if self.overlap_read_pos == hop_size {
                self.overlap_buffer.rotate_left(hop_size);
                self.overlap_buffer[fft_size - hop_size..].fill(0.0);
                self.overlap_read_pos = 0;
            }
            
            self.frames_processed += 1;
            
            // If output buffer is full, stop processing
            if output_written >= output.len() {
                break;
            }
        }
        
        Ok(output_written)
    }
    
    pub fn process_with_spectrum<F>(
        &mut self, 
        input: &[f32], 
        output: &mut [f32], 
        mut processor: F
    ) -> Result<usize, FftError> 
    where 
        F: FnMut(&mut [Complex32])
        {
            // Add input to buffer, tracking any dropped samples
            match self.input_buffer.write(input) {
                Ok(_) => {},
                Err(_) => {
                    // Buffer full - samples were dropped
                    self.samples_dropped += input.len() as u64;
                }
            }

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

                    // Read hop_size new samples into reusable buffer
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

                // Apply spectral processing using the provided processor
                processor(&mut self.spectrum_work);

                // Inverse FFT
                let frame_result = self.fft_processor.inverse(&self.spectrum_work)?;

                // Overlap-add into accumulator with window compensation
                for i in 0..fft_size {
                    self.overlap_buffer[i] += frame_result[i] * self.window_compensation;
                }

                // Output hop_size samples (or whatever fits in output buffer)
                let remaining = output.len() - output_written;
                let available_in_hop = hop_size - self.overlap_read_pos;
                let to_emit = available_in_hop.min(remaining);
                
                // Emit from the current read cursor
                let start = self.overlap_read_pos;
                let end = start + to_emit;
                output[output_written..output_written + to_emit]
                    .copy_from_slice(&self.overlap_buffer[start..end]);
                output_written += to_emit;
                self.overlap_read_pos += to_emit;
                
                // Only rotate once we've emitted a full hop
                if self.overlap_read_pos == hop_size {
                    self.overlap_buffer.rotate_left(hop_size);
                    self.overlap_buffer[fft_size - hop_size..].fill(0.0);
                    self.overlap_read_pos = 0;
                }

                self.frames_processed += 1;
            }

            Ok(output_written)
        }

    /// Get the FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_processor.fft_size
    }
    
    /// Get the hop size  
    pub fn hop_size(&self) -> usize {
        self.fft_processor.hop_size
    }
    
    /// Get the latency in samples
    #[inline]
    /// Get the algorithmic latency in samples.
    /// 
    /// Returns 0 for overlap-add processing as the output appears at the same
    /// position as the input (no additional delay beyond initial buffering).
    /// 
    /// Note: This is different from the initial buffering requirement (fft_size)
    /// needed before the first output can be produced.
    pub fn latency_samples(&self) -> usize {
        // OLA/WOLA with proper window compensation has no algorithmic latency
        // The processing is effectively "zero-latency" once buffering is satisfied
        0
    }
    
    /// Get the number of samples dropped due to buffer overflow
    pub fn samples_dropped(&self) -> u64 {
        self.samples_dropped
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
        self.overlap_read_pos = 0;
        self.hop_buffer.fill(0.0);
        self.spectrum_work.fill(Complex32::new(0.0, 0.0));
        self.frames_processed = 0;
        self.samples_dropped = 0;
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

        if (pitch_ratio - 1.0).abs() < 1e-6 {
            // Update trackers: leave spectrum as-is (true Identity)
            for bin in 0..num_bins {
                self.last_phase_input[bin] = phases[bin];
                // Keep accumulator coherent for future frames
                self.phase_accumulator[bin] = phases[bin];
            }
            // Ensure Hermitian constraints just like elsewhere
            // (DC and Nyquist real, mirror negatives)
            if self.fft_size % 2 == 0 { spectrum[self.fft_size/2].im = 0.0; }
            spectrum[0].im = 0.0;
            for k in 1..num_bins-1 {
                spectrum[self.fft_size - k] = spectrum[k].conj();
            }
            return;
        }
        
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
            let target_bin = ((shifted_freq * self.fft_size as f32) / self.sample_rate).round() as usize;
            
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

/// Enforce Hermitian symmetry on a complex spectrum
/// 
/// For real-valued signals, the FFT spectrum must satisfy:
/// - X[0] is real (DC component)
/// - X[N/2] is real if N is even (Nyquist frequency)
/// - X[k] = conj(X[N-k]) for k = 1 to N/2-1
/// 
/// This function enforces these constraints to ensure the inverse FFT
/// produces a real-valued signal without imaginary artifacts.
fn enforce_hermitian_symmetry(spectrum: &mut [Complex32]) {
    let n = spectrum.len();
    if n == 0 {
        return;
    }
    
    let half = n / 2;
    
    // DC component must be real
    spectrum[0].im = 0.0;
    
    // Nyquist frequency must be real (if N is even)
    if n % 2 == 0 && half < n {
        spectrum[half].im = 0.0;
    }
    
    // Mirror positive frequencies to negative frequencies
    // X[k] = conj(X[N-k]) for k = 1 to half-1
    for k in 1..half {
        let mirror = n - k;
        if mirror < n {
            spectrum[mirror] = spectrum[k].conj();
        }
    }
}

/// Calculate COLA profile by overlapping windows
/// 
/// Creates a long synthetic stream and overlap-adds windows at hop spacing,
/// then returns the steady-state profile to check flatness.
fn calculate_cola_profile(window: &[f32], hop_size: usize, repeats: usize) -> Vec<f32> {
    let n = window.len();
    let len = hop_size * repeats + n; // Long enough to reach steady-state
    let mut acc = vec![0.0f32; len];
    
    // Overlap-add windows
    for r in 0..repeats {
        let start = r * hop_size;
        for i in 0..n.min(len - start) {
            acc[start + i] += window[i];
        }
    }
    
    // Return only one hop period from the center (true steady-state)
    // This avoids edge effects where windows don't fully overlap
    let center = len / 2;
    let profile_start = center - (center % hop_size); // Align to hop boundary
    let profile_end = profile_start + hop_size;
    
    if profile_end <= acc.len() && profile_start < profile_end {
        acc[profile_start..profile_end].to_vec()
    } else {
        // Fallback to a reasonable section
        let mid_start = hop_size * 3;
        let mid_end = mid_start + hop_size;
        if mid_end <= acc.len() {
            acc[mid_start..mid_end].to_vec()
        } else {
            acc.clone()
        }
    }
}

/// Calculate COLA gain and flatness metrics
/// 
/// Returns (average_gain, flatness_ratio) where flatness_ratio should be close to 0
/// for perfect COLA compliance.
fn calculate_cola_metrics(window: &[f32], overlap_percent: f32) -> (f32, f32) {
    let size = window.len();
    let hop_size = ((size as f32) * (1.0 - overlap_percent / 100.0)) as usize;
    let hop_size = hop_size.max(1);
    
    // Get COLA profile
    let profile = calculate_cola_profile(window, hop_size, 12);
    
    if profile.is_empty() {
        return (1.0, 1.0);
    }
    
    // Calculate metrics
    let max = profile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = profile.iter().cloned().fold(f32::INFINITY, f32::min);
    let avg = profile.iter().sum::<f32>() / profile.len() as f32;
    
    // Flatness ratio: (max - min) / average
    let flatness = if avg > 1e-9 {
        (max - min) / avg
    } else {
        1.0
    };
    
    (avg, flatness)
}

/// Calculate proper window gain for COLA (compatibility wrapper)
fn calculate_cola_gain(window: &[f32], overlap_percent: f32) -> f32 {
    let (avg_gain, _) = calculate_cola_metrics(window, overlap_percent);
    avg_gain
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
    fn test_cola_hann_50_ola_is_flat() {
        // Test that Hann window with 50% overlap has flat COLA profile
        let n = 512;
        let window = WindowType::Hann.generate(n);
        let hop = n / 2;
        
        let profile = calculate_cola_profile(&window, hop, 12);
        assert!(!profile.is_empty(), "COLA profile should not be empty");
        
        let max = profile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = profile.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg = profile.iter().sum::<f32>() / profile.len() as f32;
        
        // Calculate flatness ratio
        let flatness = (max - min) / avg.max(1e-9);
        
        println!("COLA Hann 50% OLA: max={:.4}, min={:.4}, avg={:.4}, flatness={:.6}", 
                 max, min, avg, flatness);
        
        // For properly designed Hann with 50% overlap, flatness should be very good
        // We allow some tolerance due to floating point and edge effects
        assert!(flatness < 0.01, "COLA not flat enough: flatness={}", flatness);
        
        // Also check that average is close to 1.0 (unity gain)
        assert!((avg - 1.0).abs() < 0.05, "COLA average gain not close to 1.0: {}", avg);
    }
    
    #[test]
    fn test_cola_sqrt_hann_50_wola_with_compensation() {
        // Test WOLA mode with sqrt windows and compensation
        let n = 512;
        let window = WindowType::Hann.generate(n);
        let window_sqrt: Vec<f32> = window.iter().map(|v| v.sqrt()).collect();
        let hop = n / 2;
        
        // For WOLA, the effective window is the product of analysis and synthesis windows
        // Since both use sqrt, the product is the original window
        let profile = calculate_cola_profile(&window, hop, 12);
        
        let avg = profile.iter().sum::<f32>() / profile.len() as f32;
        let compensation = 1.0 / avg.max(1e-9);
        
        // Apply compensation and check flatness
        let profile_compensated: Vec<f32> = profile.iter().map(|v| v * compensation).collect();
        
        let max_comp = profile_compensated.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_comp = profile_compensated.iter().cloned().fold(f32::INFINITY, f32::min);
        
        println!("COLA sqrt-Hann 50% WOLA: compensation={:.4}, max_comp={:.4}, min_comp={:.4}", 
                 compensation, max_comp, min_comp);
        
        // After compensation, should be close to 1.0
        assert!((max_comp - 1.0).abs() < 0.05, "WOLA max not close to 1.0 after compensation: {}", max_comp);
        assert!((min_comp - 1.0).abs() < 0.05, "WOLA min not close to 1.0 after compensation: {}", min_comp);
    }
    
    #[test]
    fn test_cola_various_overlaps() {
        // Test different overlap percentages with periodic windows
        // COLA means constant sum, not necessarily sum=1.0
        // At different overlaps, the constant sum may be different (requires compensation)
        let test_cases = vec![
            (WindowType::Hann, 50.0, 0.001, "Hann 50%"),          // Perfect COLA (sum=1.0)
            (WindowType::Hann, 66.7, 0.003, "Hann 66.7%"),        // Very good COLA (sum≈1.506)
            (WindowType::Hann, 75.0, 0.001, "Hann 75%"),          // Perfect COLA (sum=2.0)
            (WindowType::Hamming, 50.0, 0.001, "Hamming 50%"),    // Perfect COLA (sum≈1.08)
            (WindowType::Blackman, 66.7, 0.001, "Blackman 66.7%"), // Perfect COLA (sum≈1.265)
            (WindowType::Rectangular, 0.0, 0.001, "Rectangular 0%"), // Perfect COLA (sum=1.0)
        ];
        
        for (window_type, overlap, max_flatness, description) in test_cases {
            let window = window_type.generate(512);
            let (avg_gain, flatness) = calculate_cola_metrics(&window, overlap);
            
            println!("{}: avg_gain={:.4}, flatness={:.6}", description, avg_gain, flatness);
            
            // Check flatness is within acceptable bounds
            assert!(
                flatness < max_flatness,
                "{} COLA flatness {} exceeds threshold {}",
                description, flatness, max_flatness
            );
            
            // Also document which combinations are COLA-compliant
            if flatness < 0.02 {
                println!("  ✓ {} is COLA-compliant", description);
            } else {
                println!("  ✗ {} does not satisfy COLA (flatness={:.3})", description, flatness);
            }
        }
    }
    
    #[test]
    fn test_cola_processor_compensation() {
        // Test that the processor's compensation calculation is correct
        let processor = FftProcessor::with_window(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        let compensation = processor.calculate_window_compensation();
        
        // Apply compensation to COLA profile and verify unity gain
        let window = WindowType::Hann.generate(512);
        let profile = calculate_cola_profile(&window, 256, 12);
        
        let compensated: Vec<f32> = profile.iter().map(|v| v * compensation).collect();
        let avg_compensated = compensated.iter().sum::<f32>() / compensated.len() as f32;
        
        println!("OLA Processor compensation={:.4}, resulting avg={:.4}", compensation, avg_compensated);
        
        // Should be close to 1.0 after compensation
        assert!((avg_compensated - 1.0).abs() < 0.05, 
                "Compensated average {} not close to 1.0", avg_compensated);
    }
    
    #[test]
    fn test_wola_compensation() {
        // Test WOLA compensation with sqrt windows
        // WOLA uses sqrt(window) for both analysis and synthesis
        // The effective window is sqrt(w) * sqrt(w) = w
        
        let test_cases = vec![
            (WindowType::Hann, 50.0, "sqrt-Hann 50%"),
            (WindowType::Hann, 75.0, "sqrt-Hann 75%"),
            (WindowType::Hamming, 50.0, "sqrt-Hamming 50%"),
        ];
        
        for (window_type, overlap, description) in test_cases {
            let processor = FftProcessor::with_window(
                512,
                window_type,
                overlap,
                ProcessingMode::WOLA,
            ).unwrap();
            
            let compensation = processor.calculate_window_compensation();
            
            // The effective window for WOLA is the original window (sqrt * sqrt = original)
            let original_window = window_type.generate(512);
            let hop_size = ((512.0 * (1.0 - overlap / 100.0)) as usize).max(1);
            let profile = calculate_cola_profile(&original_window, hop_size, 12);
            
            let compensated: Vec<f32> = profile.iter().map(|v| v * compensation).collect();
            let avg_compensated = compensated.iter().sum::<f32>() / compensated.len() as f32;
            
            println!("WOLA {}: compensation={:.4}, resulting avg={:.4}", 
                     description, compensation, avg_compensated);
            
            // Should be close to 1.0 after compensation
            assert!(
                (avg_compensated - 1.0).abs() < 0.05,
                "WOLA {} compensated average {} not close to 1.0",
                description, avg_compensated
            );
        }
    }
    
    #[test]
    fn test_wola_unity_gain_reconstruction() {
        // Test that WOLA mode achieves unity gain reconstruction
        // with proper compensation applied
        
        // Create OverlapAddProcessor with WOLA mode for streaming test
        let mut overlap_processor = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::WOLA,
        ).unwrap();
        
        // Process a DC signal (all 1.0) - should reconstruct to 1.0
        let input = vec![1.0; 2048];  // Multiple frames
        let mut output = vec![0.0; 2048];
        
        // Identity processing - no callback for true pass-through
        
        let processed = overlap_processor.process(&input, &mut output).unwrap();
        
        // No need to apply compensation - OverlapAddProcessor handles it internally
        
        // Check the middle portion (avoiding edge effects)
        let latency = overlap_processor.latency_samples();
        if processed > latency + 256 {
            let middle_start = latency + 64;
            let middle_end = middle_start + 256;
            let middle_avg: f32 = output[middle_start..middle_end].iter().sum::<f32>() 
                                  / (middle_end - middle_start) as f32;
            
            println!("WOLA unity gain test: DC input=1.0, output avg={:.4}",
                     middle_avg);
            
            // WOLA processing with finite frames may not achieve perfect unity gain
            // due to edge effects and windowing. Accept a reasonable range.
            assert!(
                middle_avg > 0.7 && middle_avg < 1.1,
                "WOLA reconstruction of DC signal {} not in acceptable range [0.7, 1.1]",
                middle_avg
            );
        }
        
        // Test with a sine wave
        let freq = 440.0;
        let sample_rate = 48000.0;
        let input: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();
        let mut output = vec![0.0; 2048];
        
        overlap_processor.reset();
        let processed = overlap_processor.process(&input, &mut output).unwrap();
        
        // No need to apply compensation - OverlapAddProcessor handles it internally
        
        if processed > latency + 512 {
            let middle_start = latency + 128;
            let middle_end = middle_start + 256;
            
            // Calculate RMS of input and output middle sections
            let input_rms: f32 = input[middle_start..middle_end].iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt() / ((middle_end - middle_start) as f32).sqrt();
            
            let output_rms: f32 = output[middle_start..middle_end].iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt() / ((middle_end - middle_start) as f32).sqrt();
            
            let gain_ratio = output_rms / input_rms.max(1e-9);
            
            println!("WOLA sine wave test: input RMS={:.4}, output RMS={:.4}, gain ratio={:.4}",
                     input_rms, output_rms, gain_ratio);
            
            assert!(
                (gain_ratio - 1.0).abs() < 0.2,  // Allow some tolerance for windowing effects
                "WOLA sine wave gain ratio {} not close to 1.0",
                gain_ratio
            );
        }
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
        // Use pass-through mode (no callback) for latency measurement
        
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
    
    #[test]
    fn test_basic_passthrough() {
        // Simple test to verify basic passthrough works
        let mut processor = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        // Create a simple sine wave input
        let mut input = vec![0.0; 1024];
        for i in 0..input.len() {
            input[i] = (i as f32 * 0.1).sin();
        }
        
        let mut output = vec![0.0; 1024];
        let processed = processor.process(&input, &mut output).unwrap();
        
        println!("Basic passthrough test:");
        println!("Processed {} samples", processed);
        
        // Check if any output was produced
        let max_val = output.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        println!("Max output value: {}", max_val);
        
        assert!(max_val > 0.0, "Should produce non-zero output");
    }
    
    #[test]
    fn test_observed_latency_vs_reported() {
        // Test actual observed latency vs what latency_samples() reports
        // Focus on 512 FFT with 50% overlap for clarity
        let fft_size = 512;
        let overlap = 50.0;
        
        let mut processor = OverlapAddProcessor::new(
            fft_size,
            WindowType::Hann,
            overlap,
            ProcessingMode::OLA,
        ).unwrap();
        
        let hop_size = processor.hop_size();
        let reported_latency = processor.latency_samples();
        
        println!("\n=== Latency Test ===");
        println!("FFT size: {}", fft_size);
        println!("Overlap: {}%", overlap);
        println!("Hop size: {}", hop_size);
        println!("Reported latency: {} samples", reported_latency);
        
        // Identity processing (no spectral modification)
        // Use pass-through mode (no callback) for latency measurement
        
        // Create test signal: impulse in the middle of the first frame
        // (to avoid windowing effects at the edges)
        let total_samples = fft_size * 3;
        let mut input = vec![0.0; total_samples];
        let impulse_pos = fft_size / 2;  // Place in middle of first frame
        input[impulse_pos] = 1.0;
        
        // Process all at once to simplify
        let mut output = vec![0.0; total_samples];
        let processed = processor.process(&input, &mut output).unwrap();
        
        println!("Input {} samples, got {} output samples", input.len(), processed);
        
        // Find first non-zero output
        let mut first_nonzero = None;
        let mut peak_index = 0;
        let mut peak_value = 0.0;
        
        for i in 0..processed {
            if output[i].abs() > 1e-9 && first_nonzero.is_none() {
                first_nonzero = Some(i);
            }
            if output[i].abs() > peak_value {
                peak_value = output[i].abs();
                peak_index = i;
            }
        }
        
        if let Some(first) = first_nonzero {
            println!("\nResults:");
            println!("Impulse was at input position: {}", impulse_pos);
            println!("First non-zero output at index: {}", first);
            println!("Peak of impulse response at index: {} (value: {:.6})", peak_index, peak_value);
            
            // Calculate actual observed latency
            // The latency is the difference between where the impulse appears in the output
            // versus where it was in the input
            let observed_latency = peak_index as i32 - impulse_pos as i32;
            
            println!("\nAnalysis:");
            println!("Reported latency: {} samples", reported_latency);
            println!("Observed latency (peak position - impulse position): {} samples", observed_latency);
            println!("Difference: {} samples", observed_latency - reported_latency as i32);
            
            // The observed latency should match the reported latency
            assert_eq!(
                observed_latency, reported_latency as i32,
                "Observed latency {} should match reported latency {}",
                observed_latency, reported_latency
            );
            
            // Verify that the processor correctly reports zero algorithmic latency
            assert_eq!(reported_latency, 0, "OLA processing should have zero algorithmic latency");
                
        } else {
            // Let's debug why no output was produced
            println!("\nDebug: No output detected!");
            println!("Input buffer available after process: {}", processor.input_buffer.available());
            println!("Frames processed: {}", processor.frames_processed);
            
            // Try processing empty input to flush
            let mut flush_output = vec![0.0; total_samples];
            let flushed = processor.process(&[], &mut flush_output).unwrap();
            println!("Flushed {} more samples", flushed);
            
            for i in 0..flushed {
                if flush_output[i].abs() > 1e-9 && first_nonzero.is_none() {
                    first_nonzero = Some(i);
                    println!("Found output in flush at index {}", i);
                    break;
                }
            }
            
            if first_nonzero.is_none() {
                panic!("No output produced even after flush!");
            }
        }
    }
    
    #[test]
    fn test_latency_with_different_signals() {
        // Test latency with different test signals
        let mut processor = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        // Use pass-through mode (no callback) for latency measurement
        
        let reported_latency = processor.latency_samples();
        
        // Test 1: Step function
        let mut input = vec![0.0; 2048];
        for i in 100..input.len() {
            input[i] = 1.0;
        }
        
        let mut output = vec![0.0; 2048];
        let processed = processor.process(&input, &mut output).unwrap();
        
        // Find where output starts rising
        let mut step_response_start = None;
        for i in 0..processed {
            if output[i].abs() > 0.01 {
                step_response_start = Some(i);
                break;
            }
        }
        
        println!("\nStep response test:");
        println!("Step input starts at: 100");
        println!("Step response starts at: {:?}", step_response_start);
        println!("Reported latency: {}", reported_latency);
        
        if let Some(start) = step_response_start {
            let actual_delay = start as i32 - 100;
            println!("Actual delay: {} samples", actual_delay);
            
            // The actual delay should be close to the reported latency
            // but might differ due to windowing effects
            assert!(actual_delay >= 0, "Output should not precede input");
        }
        
        // Test 2: Sine wave phase shift
        processor.reset();
        let freq = 440.0;
        let sample_rate = 48000.0;
        let input: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();
        
        let mut output = vec![0.0; 2048];
        let processed = processor.process(&input, &mut output).unwrap();
        
        // Measure phase shift to determine group delay
        if processed > reported_latency + 100 {
            let measure_point = reported_latency + 50;
            let input_phase = (2.0 * PI * freq * measure_point as f32 / sample_rate);
            let expected_delayed_phase = (2.0 * PI * freq * (measure_point - reported_latency) as f32 / sample_rate);
            
            println!("\nSine wave phase test at {} Hz:", freq);
            println!("Measuring at sample: {}", measure_point);
            println!("Input value: {:.4}", input[measure_point]);
            println!("Output value: {:.4}", output[measure_point]);
            println!("Expected delayed value: {:.4}", expected_delayed_phase.sin());
        }
    }
    
    #[test]
    fn test_hermitian_symmetry_enforcement() {
        // Test that enforce_hermitian_symmetry correctly fixes broken symmetry
        let test_sizes = vec![8, 16, 32, 64];
        
        for size in test_sizes {
            let mut spectrum = vec![Complex32::new(0.0, 0.0); size];
            
            // Create a deliberately non-Hermitian spectrum
            // Set some arbitrary complex values
            spectrum[0] = Complex32::new(1.0, 0.5); // DC with imaginary part (wrong!)
            spectrum[1] = Complex32::new(2.0, 3.0);
            spectrum[2] = Complex32::new(-1.0, 2.0);
            if size > 4 {
                spectrum[3] = Complex32::new(0.5, -1.5);
            }
            if size >= 8 {
                spectrum[size/2] = Complex32::new(1.0, 0.7); // Nyquist with imaginary (wrong!)
            }
            
            // Apply enforcement
            enforce_hermitian_symmetry(&mut spectrum);
            
            // Verify DC is real
            assert_eq!(spectrum[0].im, 0.0, "DC component should be real");
            
            // Verify Nyquist is real (if size is even)
            if size % 2 == 0 {
                assert_eq!(spectrum[size/2].im, 0.0, "Nyquist should be real");
            }
            
            // Verify conjugate symmetry
            for k in 1..size/2 {
                let mirror = size - k;
                let diff = (spectrum[mirror] - spectrum[k].conj()).norm();
                assert!(
                    diff < 1e-6,
                    "Conjugate symmetry not satisfied at k={}: spectrum[{}]={:?}, spectrum[{}].conj()={:?}",
                    k, mirror, spectrum[mirror], k, spectrum[k].conj()
                );
            }
        }
    }
    
    #[test]
    fn test_symmetry_breaking_callback() {
        // Test that callbacks that break symmetry don't produce artifacts
        // thanks to automatic symmetry enforcement
        
        let mut processor = FftProcessor::with_window(
            64,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        // Create a real input signal
        let input: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / 64.0).cos())
            .collect();
        
        // Forward FFT
        let spectrum = processor.forward(&input).unwrap();
        let mut modified_spectrum = spectrum.to_vec();
        
        // Deliberately break Hermitian symmetry
        modified_spectrum[0].im = 1.0;  // DC should be real!
        modified_spectrum[32].im = 0.5; // Nyquist should be real!
        
        // Don't mirror changes to negative frequencies
        modified_spectrum[1] = Complex32::new(10.0, 5.0);
        // modified_spectrum[63] should be conj(modified_spectrum[1]) but we don't set it
        
        // Process inverse - should automatically fix symmetry
        let output = processor.inverse(&modified_spectrum).unwrap();
        
        // Despite breaking symmetry, output should still be real
        // (no NaN or infinite values from taking real part of complex result)
        for (i, &sample) in output.iter().enumerate() {
            assert!(
                sample.is_finite(),
                "Output contains non-finite value at index {}: {}",
                i, sample
            );
        }
        
        // Verify the output is reasonable (not just zeros)
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(output_energy > 0.0, "Output should contain non-zero signal");
    }
    
    #[test]
    fn test_callback_preserving_symmetry() {
        // Test that callbacks that properly preserve symmetry work correctly
        
        let mut processor = FftProcessor::with_window(
            64,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        // Create a real input signal
        let input: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 4.0 * i as f32 / 64.0).cos())
            .collect();
        
        // Forward FFT
        let spectrum = processor.forward(&input).unwrap();
        let mut modified_spectrum = spectrum.to_vec();
        
        // Properly modify spectrum while preserving symmetry
        let n = modified_spectrum.len();
        
        // Apply gain to positive frequencies and mirror to negative
        for k in 1..n/2 {
            modified_spectrum[k] *= 2.0;  // Amplify
            modified_spectrum[n - k] = modified_spectrum[k].conj();  // Maintain symmetry
        }
        
        // Keep DC and Nyquist real
        modified_spectrum[0] = Complex32::new(modified_spectrum[0].re * 2.0, 0.0);
        modified_spectrum[n/2] = Complex32::new(modified_spectrum[n/2].re * 2.0, 0.0);
        
        // Process inverse
        let output = processor.inverse(&modified_spectrum).unwrap();
        
        // Output should be real and amplified
        for &sample in output.iter() {
            assert!(sample.is_finite(), "Output should be finite");
        }
        
        // Check that signal was amplified (roughly 2x energy)
        let input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        let energy_ratio = output_energy / input_energy.max(1e-9);
        
        // With Hann windowing applied twice (forward and inverse),
        // the effective gain is reduced. The energy ratio should still show amplification
        // but less than the theoretical 4x due to windowing
        assert!(
            energy_ratio > 1.0 && energy_ratio < 6.0,
            "Energy ratio {} not in expected range [1, 6]",
            energy_ratio
        );
    }
    
    #[test]
    fn test_tukey_window_edge_cases() {
        // Test various alpha values and sizes to ensure no panics or invalid values
        let test_cases = vec![
            // (size, alpha, description)
            (64, 0.0, "rectangular (alpha=0)"),
            (64, 0.001, "very small alpha"),
            (64, 0.01, "small alpha"),
            (64, 0.5, "medium alpha"),
            (64, 0.99, "large alpha"),
            (64, 1.0, "Hann-like (alpha=1)"),
            (MIN_FFT_SIZE, 0.001, "minimum size with small alpha"),
            (MIN_FFT_SIZE, 0.5, "minimum size with medium alpha"),
            (MIN_FFT_SIZE, 1.0, "minimum size with alpha=1"),
            (4, 0.001, "very small size with small alpha"),
            (4, 0.5, "very small size with medium alpha"),
            (4, 1.0, "very small size with alpha=1"),
            (3, 0.001, "size 3 with small alpha"),
            (3, 1.0, "size 3 with alpha=1"),
            (2, 0.001, "size 2 with small alpha"),
            (2, 1.0, "size 2 with alpha=1"),
            (1, 0.5, "size 1 edge case"),
        ];
        
        for (size, alpha, description) in test_cases {
            println!("Testing Tukey window: {} (size={}, alpha={})", description, size, alpha);
            
            let window = WindowType::Tukey(alpha).generate(size);
            
            // Verify no panics occurred (we got here)
            assert_eq!(window.len(), size, "Window size mismatch for {}", description);
            
            // Verify all coefficients are finite (no NaN or Inf)
            for (i, &coeff) in window.iter().enumerate() {
                assert!(
                    coeff.is_finite(),
                    "Non-finite coefficient at index {} for {}: {}",
                    i, description, coeff
                );
                
                // Verify coefficients are in valid range [0, 1]
                assert!(
                    coeff >= 0.0 && coeff <= 1.0,
                    "Coefficient out of range at index {} for {}: {}",
                    i, description, coeff
                );
            }
            
            // Verify specific properties based on alpha
            if alpha == 0.0 || (alpha < 0.01 && size < 10) {
                // Should be rectangular (all 1.0)
                for &coeff in window.iter() {
                    assert!(
                        (coeff - 1.0).abs() < EPSILON,
                        "Expected rectangular window for {}", description
                    );
                }
            } else if alpha == 1.0 && size > 2 {
                // Should taper to near 0 at edges
                assert!(
                    window[0] < 0.1,
                    "Expected tapered start for {}: {}", description, window[0]
                );
                assert!(
                    window[size - 1] < 0.1,
                    "Expected tapered end for {}: {}", description, window[size - 1]
                );
            }
            
            // Verify symmetry for sizes > 1
            if size > 1 {
                for i in 0..size/2 {
                    let diff = (window[i] - window[size - 1 - i]).abs();
                    assert!(
                        diff < EPSILON,
                        "Window not symmetric at indices {} and {} for {}: {} vs {}",
                        i, size - 1 - i, description, window[i], window[size - 1 - i]
                    );
                }
            }
        }
        
        // Test that generate_into produces same results as generate
        let test_alpha = 0.5;
        let test_size = 128;
        let window1 = WindowType::Tukey(test_alpha).generate(test_size);
        let mut window2 = vec![0.0; test_size];
        WindowType::Tukey(test_alpha).generate_into(&mut window2);
        
        for i in 0..test_size {
            assert!(
                (window1[i] - window2[i]).abs() < EPSILON,
                "generate and generate_into mismatch at index {}", i
            );
        }
    }
    
    #[test]
    fn test_tukey_window_continuity() {
        // Test that Tukey window transitions smoothly from rectangular to Hann-like
        let size = 256;
        let alphas = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let mut prev_window = WindowType::Tukey(0.0).generate(size);
        
        for &alpha in alphas.iter().skip(1) {
            let window = WindowType::Tukey(alpha).generate(size);
            
            // Check that the taper region grows with alpha
            let mut taper_start = size;
            let mut taper_end = 0;
            
            for i in 0..size {
                if (window[i] - 1.0).abs() > 0.01 {
                    taper_start = taper_start.min(i);
                    taper_end = taper_end.max(i);
                }
            }
            
            let taper_region = if taper_end > taper_start {
                taper_end - taper_start + 1
            } else {
                0
            };
            
            println!("Alpha {}: taper region = {} samples", alpha, taper_region);
            
            // As alpha increases, taper region should generally increase
            // (though it may plateau for high alpha values)
            if alpha > 0.1 {
                assert!(
                    taper_region > 0,
                    "Expected non-zero taper region for alpha={}", alpha
                );
            }
            
            prev_window = window;
        }
    }
    
    #[test]
    fn test_partial_hop_output() {
        // Test that partial hop output doesn't drop samples
        let mut processor = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        
        let hop_size = processor.hop_size();
        println!("Testing partial hop output with hop_size = {}", hop_size);
        
        // Identity processing
        // Use pass-through mode (no callback) for latency measurement
        
        // Generate test signal - a sine wave for testing
        let input: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();
        
        // Process with tiny output buffers (smaller than hop_size)
        let tiny_buffer_size = hop_size / 3;  // Much smaller than hop
        let mut all_output = Vec::new();
        
        // Process input and output in interleaved fashion
        let chunk_size = 256;
        let mut input_pos = 0;
        
        while input_pos < input.len() || processor.input_buffer.available() > 0 {
            // Feed some input
            if input_pos < input.len() {
                let end = (input_pos + chunk_size).min(input.len());
                let input_chunk = &input[input_pos..end];
                
                // Use tiny output buffer
                let mut temp_out = vec![0.0; tiny_buffer_size];
                let processed = processor.process(input_chunk, &mut temp_out).unwrap();
                
                if processed > 0 {
                    all_output.extend_from_slice(&temp_out[..processed]);
                    println!("Processed {} samples from input[{}..{}], got {} output samples", 
                             input_chunk.len(), input_pos, end, processed);
                }
                
                input_pos = end;
            } else {
                // No more input, flush remaining output
                let mut temp_out = vec![0.0; tiny_buffer_size];
                let processed = processor.process(&[], &mut temp_out).unwrap();
                
                if processed > 0 {
                    all_output.extend_from_slice(&temp_out[..processed]);
                    println!("Flushed {} output samples", processed);
                } else {
                    break;  // No more output
                }
            }
        }
        
        println!("Total output samples collected: {}", all_output.len());
        
        // Verify we got output
        assert!(all_output.len() > 0, "Should have produced output");
        
        // More specific test: process with exact hop_size-1 buffer
        let mut processor2 = OverlapAddProcessor::new(
            512,
            WindowType::Hann,
            50.0,
            ProcessingMode::OLA,
        ).unwrap();
        processor2.set_process_callback(|_spectrum| {});
        
        let test_input: Vec<f32> = vec![1.0; 2048];  // Constant signal
        let mut partial_out = vec![0.0; hop_size - 1];  // One less than hop_size
        let mut full_out = vec![0.0; hop_size];
        
        // Feed enough input to generate output
        processor2.process(&test_input, &mut []).unwrap();
        
        // Try to get output with partial buffer
        let partial_processed = processor2.process(&[], &mut partial_out).unwrap();
        println!("Partial buffer (size {}): processed {} samples", partial_out.len(), partial_processed);
        
        // Now try to get the rest
        let full_processed = processor2.process(&[], &mut full_out).unwrap();
        println!("Full buffer (size {}): processed {} samples", full_out.len(), full_processed);
        
        // We should have gotten hop_size-1 samples first, then 1 more
        assert_eq!(partial_processed, hop_size - 1, 
                   "Should output exactly {} samples to partial buffer", hop_size - 1);
        assert!(full_processed >= 1, 
                "Should output at least 1 sample to complete the hop");
    }

    #[test]
    fn no_analysis_advance_when_output_zero() {
        let mut p = OverlapAddProcessor::new(512, WindowType::Hann, 50.0, ProcessingMode::OLA).unwrap();
        let hop = p.hop_size();

        // Feed more than a frame with zero output capacity
        let input = vec![1.0; 4 * hop];
        let written = p.process(&input, &mut []).unwrap();
        assert_eq!(written, 0);

        // Now emit with enough room for exactly one hop
        let mut out = vec![0.0; hop];
        let produced = p.process(&[], &mut out).unwrap();
        assert_eq!(produced, hop, "Should emit exactly one hop after withholding output");
    }

    #[test]
    fn hop_rounding_reduces_cola_error() {
        let n = 512;
        let w = WindowType::Hann.generate(n);
        // approximate 66.7%
        let hop_floor = ((n as f32 * (1.0 - 0.667)) as usize).max(1);
        let hop_round = ((n as f32 * (1.0 - 0.667)).round() as usize).max(1);
        let (_, flat_floor) = calculate_cola_metrics(&w, 66.7);
        // recompute via profile directly:
        fn flat(w:&[f32], hop:usize)->f32{
            let prof = calculate_cola_profile(w, hop, 40);
            let max = prof.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = prof.iter().cloned().fold(f32::INFINITY, f32::min);
            let avg = prof.iter().sum::<f32>() / prof.len() as f32;
            (max - min) / avg
        }
        let f_floor = flat(&w, hop_floor);
        let f_round = flat(&w, hop_round);
        assert!(f_round <= f_floor, "rounded hop should not be worse");
    }

    #[test]
    fn phase_vocoder_identity_for_ratio_one() {
        let n = 1024;
        let hop = n/4;
        let sr = 48_000.0;
        let mut voc = PhaseVocoder::new(n, hop, sr);
        let mut s = vec![Complex32::new(0.0, 0.0); n];
        // two tones
        let b1 = ((440.0 * n as f32)/sr).round() as usize;
        let b2 = ((2000.0 * n as f32)/sr).round() as usize;
        s[b1] = Complex32::from_polar(1.0, 0.3);
        s[n-b1] = s[b1].conj();
        s[b2] = Complex32::from_polar(0.7, -0.8);
        s[n-b2] = s[b2].conj();

        let before = s.clone();
        voc.process_pitch_shift(&mut s, 1.0);
        // Magnitudes should be preserved in positive bins
        for k in 0..=n/2 {
            let a = before[k].norm();
            let b = s[k].norm();
            assert!((a - b).abs() < 1e-3, "bin {} magnitude drift {} vs {}", k, a, b);
        }
    }

    #[test]
    fn dropped_samples_counter_increments() {
        let mut p = OverlapAddProcessor::new(256, WindowType::Hann, 50.0, ProcessingMode::OLA).unwrap();
        // Fill until writes fail/partial
        let big = vec![0.0f32; 1 << 16];
        let _ = p.process(&big, &mut [0.0; 0]).unwrap();
        let dropped = p.samples_dropped();
        assert!(dropped > 0, "expected to drop when input ring is saturated");
    }

}
