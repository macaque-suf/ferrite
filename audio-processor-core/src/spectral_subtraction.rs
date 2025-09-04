//! Spectral subtraction for noise reduction
//! 
//! This module implements the spectral subtraction algorithm for removing
//! stationary noise from audio signals. It works by estimating the noise
//! spectrum during silence periods and subtracting it from the noisy signal
//! spectrum in the frequency domain.
//!
//! Based on the classical Boll (1979) algorithm with modern enhancements
//! for reduced musical noise and improved quality.

use crate::fft_processor::{FftError, OverlapAddProcessor, ProcessingMode, WindowType};
use crate::noise_profile::{NoiseProfile, AdaptiveNoiseTracker, NoiseProfileError, ComfortNoiseGenerator};
use crate::spectral_smoothing::{SpectralSmoother, SmoothingType};
use crate::utils::{db_to_linear, linear_to_db, ParameterSmoother, DENORMAL_PREVENTION};
use num_complex::Complex32;
use std::f32::consts::PI;

// ============================================================================
// Constants
// ============================================================================

/// Default over-subtraction factor
pub const DEFAULT_ALPHA: f32 = 2.0;

/// Default spectral floor (prevents musical noise) - 0.02 is approximately -34 dB
/// Lowered from 0.1 (-20 dB) for cleaner speech with 2-D smoothing
pub const DEFAULT_BETA: f32 = 0.02;

/// Default FFT size for spectral processing
/// Changed to 256 to match Web Audio's 128-sample callback (256 * 0.5 = 128 hop size)
pub const DEFAULT_FFT_SIZE: usize = 256;

/// Default window overlap percentage
pub const DEFAULT_OVERLAP_PERCENT: f32 = 50.0;

/// Default noise estimation mode
pub const DEFAULT_ADAPTIVE_MODE: bool = true;

/// Default VAD threshold in dB
pub const DEFAULT_VAD_THRESHOLD_DB: f32 = 6.0;

/// Maximum over-subtraction factor
pub const MAX_ALPHA: f32 = 5.0;

/// Minimum spectral floor
pub const MIN_BETA: f32 = 0.001;

/// Gain smoothing time in ms
pub const GAIN_SMOOTHING_MS: f32 = 5.0;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingError {
    /// Invalid configuration parameters
    InvalidConfiguration(String),
    
    /// Buffer size mismatch
    BufferSizeMismatch { expected: usize, actual: usize },
    
    /// FFT processing error
    FFTError(String),
    
    /// Noise profile error
    NoiseProfileError(String),
    
    /// Not initialized
    NotInitialized,
    
    /// Processing failure
    ProcessingFailed(String),
}

impl std::fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingError::InvalidConfiguration(msg) => 
                write!(f, "Invalid configuration: {}", msg),
            ProcessingError::BufferSizeMismatch { expected, actual } => 
                write!(f, "Buffer size mismatch: expected {}, got {}", expected, actual),
            ProcessingError::FFTError(msg) => 
                write!(f, "FFT error: {}", msg),
            ProcessingError::NoiseProfileError(msg) => 
                write!(f, "Noise profile error: {}", msg),
            ProcessingError::NotInitialized => 
                write!(f, "Spectral subtractor not initialized"),
            ProcessingError::ProcessingFailed(msg) => 
                write!(f, "Processing failed: {}", msg),
        }
    }
}

impl std::error::Error for ProcessingError {}

impl From<FftError> for ProcessingError {
    fn from(err: FftError) -> Self {
        ProcessingError::FFTError(err.to_string())
    }
}

impl From<NoiseProfileError> for ProcessingError {
    fn from(err: NoiseProfileError) -> Self {
        ProcessingError::NoiseProfileError(err.to_string())
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for spectral subtraction
#[derive(Debug, Clone)]
pub struct SpectralSubtractionConfig {
    /// Over-subtraction factor (typically 1.0-3.0)
    pub alpha: f32,
    
    /// Spectral floor (typically 0.01-0.1) to prevent musical noise
    pub beta: f32,
    
    /// FFT size (must be power of 2)
    pub fft_size: usize,
    
    /// Window type for analysis
    pub window_type: WindowType,
    
    /// Overlap percentage (0-95)
    pub overlap_percent: f32,
    
    /// Sample rate in Hz
    pub sample_rate: f32,
    
    /// Enable adaptive noise estimation
    pub adaptive_mode: bool,
    
    /// VAD threshold in dB for speech detection
    pub vad_threshold_db: f32,
    
    /// Processing mode (OLA or WOLA)
    pub processing_mode: ProcessingMode,
    
    /// Enable gain smoothing to reduce artifacts
    pub enable_smoothing: bool,
    
    
    /// Enable comfort noise injection
    pub enable_comfort_noise: bool,
    
    /// Comfort noise gain (0.0-1.0)
    pub comfort_noise_gain: f32,
    
    /// Use Wiener filter mode (alternative to basic subtraction)
    pub wiener_filter_mode: bool,
    
    /// Non-linear subtraction exponent (1.0 = linear, 2.0 = power)
    pub subtraction_exponent: f32,
}

impl Default for SpectralSubtractionConfig {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            fft_size: DEFAULT_FFT_SIZE,
            window_type: WindowType::Hann,
            overlap_percent: DEFAULT_OVERLAP_PERCENT,
            sample_rate: 48000.0,
            adaptive_mode: DEFAULT_ADAPTIVE_MODE,
            vad_threshold_db: DEFAULT_VAD_THRESHOLD_DB,
            processing_mode: ProcessingMode::OLA,
            enable_smoothing: true,
            enable_comfort_noise: false,
            comfort_noise_gain: 0.05,
            wiener_filter_mode: false,
            subtraction_exponent: 1.0,
        }
    }
}

impl SpectralSubtractionConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), ProcessingError> {
        if self.alpha < 0.0 || self.alpha > MAX_ALPHA {
            return Err(ProcessingError::InvalidConfiguration(
                format!("Alpha {} out of range [0, {}]", self.alpha, MAX_ALPHA)
            ));
        }
        
        if self.beta < MIN_BETA || self.beta > 1.0 {
            return Err(ProcessingError::InvalidConfiguration(
                format!("Beta {} out of range [{}, 1.0]", self.beta, MIN_BETA)
            ));
        }
        
        if !self.fft_size.is_power_of_two() || self.fft_size < 64 || self.fft_size > 8192 {
            return Err(ProcessingError::InvalidConfiguration(
                format!("FFT size {} must be power of 2 between 64 and 8192", self.fft_size)
            ));
        }
        
        if self.overlap_percent < 0.0 || self.overlap_percent > 95.0 {
            return Err(ProcessingError::InvalidConfiguration(
                format!("Overlap {} out of range [0, 95]", self.overlap_percent)
            ));
        }
        
        if self.sample_rate <= 0.0 {
            return Err(ProcessingError::InvalidConfiguration(
                format!("Invalid sample rate: {}", self.sample_rate)
            ));
        }
        
        if self.subtraction_exponent < 0.5 || self.subtraction_exponent > 2.0 {
            return Err(ProcessingError::InvalidConfiguration(
                format!("Subtraction exponent {} out of range [0.5, 2.0]", self.subtraction_exponent)
            ));
        }
        
        Ok(())
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics from spectral subtraction processing
#[derive(Debug, Clone, Default)]
pub struct SubtractionStats {
    /// Total frames processed
    pub frames_processed: u64,
    
    /// Frames classified as speech
    pub speech_frames: u64,
    
    /// Frames classified as noise
    pub noise_frames: u64,
    
    /// Average noise reduction in dB
    pub avg_noise_reduction_db: f32,
    
    /// Peak noise reduction in dB
    pub peak_noise_reduction_db: f32,
    
    /// Current SNR estimate in dB
    pub current_snr_db: f32,
}

// ============================================================================
// Main Spectral Subtractor
// ============================================================================

/// Spectral subtractor for noise reduction
pub struct SpectralSubtractor {
    // Core components
    fft_processor: OverlapAddProcessor,
    noise_tracker: AdaptiveNoiseTracker,
    config: SpectralSubtractionConfig,
    
    // Buffers for processing
    magnitude_buffer: Vec<f32>,
    phase_buffer: Vec<f32>,
    gain_buffer: Vec<f32>,
    
    // Smoothing
    gain_smoothers: Vec<ParameterSmoother>,
    
    // Sample buffering for single-sample processing
    sample_buffer: Vec<f32>,
    sample_buffer_index: usize,
    output_buffer: Vec<f32>,
    output_buffer_index: usize,
    
    // Comfort noise generator
    comfort_noise_gen: Option<ComfortNoiseGenerator>,
    
    // State
    initialized: bool,
    noise_profile_frozen: bool,
    stats: SubtractionStats,
    
    // DSP Improvement: Decision-directed a priori SNR estimation
    a_priori_snr: Vec<f32>,
    prev_output_power: Vec<f32>,
    
    // DSP Improvement: Frequency-dependent parameters
    alpha_freq: Vec<f32>,  // Frequency-dependent over-subtraction
    beta_freq: Vec<f32>,   // Frequency-dependent spectral floor
    
    // DSP Improvement: Speech Presence Probability (SPP)
    speech_presence_prob: Vec<f32>,  // Per-bin speech presence probability
    spp_smoothing_factor: f32,        // Smoothing factor for SPP estimation
    
    // Performance optimization: Reusable scratch buffers
    noise_spectrum_buf: Vec<f32>,     // Buffer for noise spectrum (avoid to_vec())
    comfort_mag_buf: Vec<f32>,        // Buffer for comfort noise magnitudes
    comfort_phase_buf: Vec<f32>,      // Buffer for comfort noise phases
    smoothed_gains_buf: Vec<f32>,     // Buffer for spectral smoothing
    
    // Spectral smoother to reduce musical noise
    spectral_smoother: SpectralSmoother,
}

impl SpectralSubtractor {
    /// Create a new spectral subtractor with configuration
    pub fn new(config: SpectralSubtractionConfig) -> Result<Self, ProcessingError> {
        config.validate()?;
        
        // Create FFT processor
        let fft_processor = OverlapAddProcessor::new(
            config.fft_size,
            config.window_type,
            config.overlap_percent,
            config.processing_mode,
        )?;
        
        // Calculate number of frequency bins
        let num_bins = config.fft_size / 2 + 1;
        
        // Create noise tracker
        let noise_tracker = AdaptiveNoiseTracker::new(
            num_bins,
            config.sample_rate,
            config.fft_size,
        );
        
        // Create gain smoothers for each frequency bin
        // Fix: Use frames per second instead of samples per second since smoothers
        // are called once per FFT hop, not once per sample
        let frames_per_second = config.sample_rate / fft_processor.hop_size() as f32;
        let gain_smoothers = (0..num_bins)
            .map(|_| ParameterSmoother::new(1.0, GAIN_SMOOTHING_MS, frames_per_second))
            .collect();
        
        // Create comfort noise generator if enabled
        let comfort_noise_gen = if config.enable_comfort_noise {
            let mut gen = ComfortNoiseGenerator::new(num_bins);
            gen.set_gain(config.comfort_noise_gain);
            Some(gen)
        } else {
            None
        };
        
        // Initialize frequency-dependent parameters
        let mut alpha_freq = vec![config.alpha; num_bins];
        let mut beta_freq = vec![config.beta; num_bins];
        
        // Calculate frequency bins in Hz
        let freq_per_bin = config.sample_rate / config.fft_size as f32;
        
        // Apply frequency-dependent scaling
        // Adapt speech band based on sample rate: for 48kHz use wider band (200-7000 Hz)
        let speech_low = if config.sample_rate >= 44100.0 { 200.0 } else { 300.0 };
        let speech_high = if config.sample_rate >= 44100.0 { 7000.0 } else { 3400.0 };
        
        for i in 0..num_bins {
            let freq_hz = i as f32 * freq_per_bin;
            
            // Alpha: Less aggressive in speech formant regions
            if freq_hz >= speech_low && freq_hz <= speech_high {
                alpha_freq[i] = config.alpha;
            } else {
                alpha_freq[i] = config.alpha * 1.5; // More aggressive outside speech band
            }
            
            // Beta: Smaller floor in speech regions for clarity
            if freq_hz >= speech_low && freq_hz <= speech_high {
                beta_freq[i] = config.beta * 0.5; // Lower floor in speech band
            } else if freq_hz < 100.0 || freq_hz > 8000.0 {
                beta_freq[i] = config.beta * 2.0; // Higher floor at extremes
            } else {
                beta_freq[i] = config.beta;
            }
        }
        
        Ok(Self {
            fft_processor,
            noise_tracker,
            config: config.clone(),
            magnitude_buffer: vec![0.0; num_bins],
            phase_buffer: vec![0.0; num_bins],
            gain_buffer: vec![1.0; num_bins],
            gain_smoothers,
            sample_buffer: vec![0.0; config.fft_size],
            sample_buffer_index: 0,
            output_buffer: vec![0.0; config.fft_size * 2],
            output_buffer_index: 0,
            comfort_noise_gen,
            initialized: true,
            noise_profile_frozen: false,
            stats: SubtractionStats::default(),
            a_priori_snr: vec![1.0; num_bins],
            prev_output_power: vec![0.0; num_bins],
            alpha_freq,
            beta_freq,
            speech_presence_prob: vec![0.0; num_bins],
            spp_smoothing_factor: 0.9,
            noise_spectrum_buf: vec![0.0; num_bins],
            comfort_mag_buf: vec![0.0; num_bins],
            comfort_phase_buf: vec![0.0; num_bins],
            smoothed_gains_buf: vec![0.0; num_bins],
            // Initialize with median filter by default for musical noise reduction
            spectral_smoother: SpectralSmoother::new(
                SmoothingType::Median { window_size: 5 },
                num_bins
            ),
        })
    }
    
    /// Create with default configuration and specified sample rate
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        let mut config = SpectralSubtractionConfig::default();
        config.sample_rate = sample_rate;
        Self::new(config).expect("Default config should be valid")
    }
    
    /// Core spectral subtraction processing (optimized to work directly on complex spectrum)
    fn process_spectrum(&mut self, spectrum: &mut [Complex32]) {
        let num_bins = self.magnitude_buffer.len();
        
        // Extract magnitude only (no phase needed with direct complex multiplication)
        for i in 0..num_bins.min(spectrum.len()) {
            self.magnitude_buffer[i] = spectrum[i].norm();
        }
        
        // Copy noise spectrum to reusable buffer (avoid allocation)
        self.noise_spectrum_buf.copy_from_slice(self.noise_tracker.profile().spectrum());
        
        // Update noise profile if adaptive mode
        let is_speech = if self.config.adaptive_mode && !self.noise_profile_frozen {
            self.noise_tracker.process(&self.magnitude_buffer)
        } else {
            self.noise_tracker.profile().is_speech(&self.magnitude_buffer, Some(self.config.vad_threshold_db))
        };
        
        // Apply spectral subtraction based on mode
        if self.config.wiener_filter_mode {
            self.apply_wiener_filter_indexed(is_speech);
        } else {
            self.apply_spectral_subtraction_indexed(is_speech);
        }
        
        // Apply comfort noise if enabled (still needs phase for noise generation)
        if self.config.enable_comfort_noise && !is_speech {
            // Extract phase only when needed for comfort noise
            for i in 0..num_bins.min(spectrum.len()) {
                self.phase_buffer[i] = spectrum[i].arg();
            }
            self.apply_comfort_noise();
        }
        
        // Apply gain smoothing if enabled
        if self.config.enable_smoothing {
            self.apply_gain_smoothing();
        }
        
        // Apply gains directly to complex spectrum (avoiding from_polar)
        for i in 0..num_bins.min(spectrum.len()) {
            // Multiply complex number by real gain - preserves phase automatically
            spectrum[i] *= self.gain_buffer[i];
        }
        
        // Maintain conjugate symmetry for real-valued output
        if spectrum.len() == self.config.fft_size {
            for i in 1..num_bins - 1 {
                if i < spectrum.len() && self.config.fft_size - i < spectrum.len() {
                    spectrum[self.config.fft_size - i] = spectrum[i].conj();
                }
            }
            // Zero imaginary parts of DC and Nyquist for numerical hygiene
            spectrum[0].im = 0.0;
            if num_bins > 1 && num_bins - 1 < spectrum.len() {
                spectrum[num_bins - 1].im = 0.0;
            }
        }
        
        // Update statistics
        self.update_statistics(is_speech);
    }
    
    /// Apply basic spectral subtraction (indexed version using internal buffer)
    fn apply_spectral_subtraction_indexed(&mut self, is_speech: bool) {
        let exponent = self.config.subtraction_exponent;
        const ALPHA_XI: f32 = 0.98;
        
        for i in 0..self.magnitude_buffer.len().min(self.noise_spectrum_buf.len()) {
            let signal_power = self.magnitude_buffer[i].powf(exponent);
            let noise_power = self.noise_spectrum_buf[i].powf(exponent);
            
            // Decision-directed a priori SNR estimation
            let posterior_snr = (signal_power / (noise_power + DENORMAL_PREVENTION)).max(0.0);
            
            // First term: smoothed gain from previous frame
            let prev_gain_term = if self.prev_output_power[i] > 0.0 {
                ALPHA_XI * (self.prev_output_power[i] / (noise_power + DENORMAL_PREVENTION))
            } else {
                0.0
            };
            
            // Second term: ML estimate from current frame
            let ml_term = (1.0 - ALPHA_XI) * (posterior_snr - 1.0).max(0.0);
            
            // Combined a priori SNR
            self.a_priori_snr[i] = prev_gain_term + ml_term;
            
            // Adaptive alpha based on a priori SNR
            let mut alpha = if self.a_priori_snr[i] > 10.0 {
                self.alpha_freq[i] * 0.7  // Less aggressive when SNR is high
            } else if self.a_priori_snr[i] < 1.0 {
                self.alpha_freq[i] * 1.3  // More aggressive when SNR is low
            } else {
                self.alpha_freq[i]
            };
            
            // Further adjust alpha based on speech detection
            if !is_speech {
                alpha *= 1.2;
            }
            
            // Use frequency-dependent beta
            let beta = self.beta_freq[i];
            
            // Spectral subtraction with frequency-dependent parameters
            let subtracted_power = signal_power - alpha * noise_power;
            let floored_power = (beta * noise_power).max(0.0);
            
            let output_power = if subtracted_power > floored_power {
                subtracted_power
            } else {
                floored_power
            };
            
            // Store output power for next frame's a priori SNR estimation
            self.prev_output_power[i] = output_power;
            
            // Calculate gain
            let ratio = (output_power / (signal_power + DENORMAL_PREVENTION)).max(0.0);
            let mut g = ratio.powf(1.0 / exponent);
            
            // Clamp to [floor, 1.0]
            g = g.min(1.0);
            
            // Use frequency-dependent gain floor
            let gain_floor = beta.min(1.0);
            g = g.max(gain_floor);
            
            self.gain_buffer[i] = g;
        }
    }
    
    /// Apply basic spectral subtraction with frequency-dependent parameters
    fn apply_spectral_subtraction(&mut self, noise_spectrum: &[f32], is_speech: bool) {
        let exponent = self.config.subtraction_exponent;
        const ALPHA_XI: f32 = 0.98; // Decision-directed smoothing factor
        
        for i in 0..self.magnitude_buffer.len().min(noise_spectrum.len()) {
            let signal_power = self.magnitude_buffer[i].powf(exponent);
            let noise_power = noise_spectrum[i].powf(exponent);
            
            // Decision-directed a priori SNR estimation
            let posterior_snr = (signal_power / (noise_power + DENORMAL_PREVENTION)).max(0.0);
            
            // First term: smoothed gain from previous frame
            let prev_gain_term = if self.prev_output_power[i] > 0.0 {
                ALPHA_XI * (self.prev_output_power[i] / (noise_power + DENORMAL_PREVENTION))
            } else {
                0.0
            };
            
            // Second term: ML estimate from current frame
            let ml_term = (1.0 - ALPHA_XI) * (posterior_snr - 1.0).max(0.0);
            
            // Combined a priori SNR
            self.a_priori_snr[i] = prev_gain_term + ml_term;
            
            // Adaptive alpha based on a priori SNR
            let mut alpha = if self.a_priori_snr[i] > 10.0 {
                self.alpha_freq[i] * 0.7  // Less aggressive when SNR is high
            } else if self.a_priori_snr[i] < 1.0 {
                self.alpha_freq[i] * 1.3  // More aggressive when SNR is low
            } else {
                self.alpha_freq[i]
            };
            
            // Further adjust alpha based on speech detection
            if !is_speech {
                alpha *= 1.2;
            }
            
            // Use frequency-dependent beta
            let beta = self.beta_freq[i];
            
            // Spectral subtraction with frequency-dependent parameters
            let subtracted_power = signal_power - alpha * noise_power;
            let floored_power = (beta * noise_power).max(0.0);
            
            let output_power = if subtracted_power > floored_power {
                subtracted_power
            } else {
                floored_power
            };
            
            // Store output power for next frame's a priori SNR estimation
            self.prev_output_power[i] = output_power;
            
            // Calculate gain
            let ratio = (output_power / (signal_power + DENORMAL_PREVENTION)).max(0.0);
            let mut g = ratio.powf(1.0 / exponent);
            
            // Clamp to [floor, 1.0]
            g = g.min(1.0);
            
            // Use frequency-dependent gain floor
            let gain_floor = beta.min(1.0);
            g = g.max(gain_floor);
            
            self.gain_buffer[i] = g;
            // Don't modify magnitude_buffer - we'll apply gain directly to complex spectrum
        }
    }
    
    /// Apply Wiener filter (indexed version using internal buffer)
    fn apply_wiener_filter_indexed(&mut self, is_speech: bool) {
        // First calculate SPP for each frequency bin
        self.calculate_speech_presence_probability_indexed(is_speech);
        
        for i in 0..self.magnitude_buffer.len().min(self.noise_spectrum_buf.len()) {
            let signal_power = self.magnitude_buffer[i] * self.magnitude_buffer[i];
            let noise_power = self.noise_spectrum_buf[i] * self.noise_spectrum_buf[i];
            
            // Wiener gain formula
            let snr = signal_power / (noise_power + DENORMAL_PREVENTION);
            let wiener_gain = snr / (1.0 + snr);
            
            // SPP-based blending: g = p * g_wiener + (1-p) * g_floor
            let spp = self.speech_presence_prob[i];
            let floor_gain = self.beta_freq[i];
            
            // Blend between Wiener gain and spectral floor based on SPP
            self.gain_buffer[i] = spp * wiener_gain + (1.0 - spp) * floor_gain;
            
            // Ensure gain is within bounds
            self.gain_buffer[i] = self.gain_buffer[i].clamp(floor_gain, 1.0);
        }
    }
    
    /// Calculate SPP (indexed version using internal buffer)
    fn calculate_speech_presence_probability_indexed(&mut self, global_is_speech: bool) {
        for i in 0..self.magnitude_buffer.len().min(self.noise_spectrum_buf.len()) {
            // Use a priori SNR for SPP estimation
            let xi = self.a_priori_snr[i];
            
            // SPP estimation using sigmoid function
            let spp_instant = if xi > 0.0 {
                let xi_db = 10.0 * xi.log10();
                let threshold_db = 3.0;
                let k = 0.3;
                1.0 / (1.0 + ((-k * (xi_db - threshold_db)).exp()))
            } else {
                0.0
            };
            
            // Apply global VAD decision as a factor
            let vad_factor = if global_is_speech { 1.2 } else { 0.8 };
            let spp_instant = (spp_instant * vad_factor).clamp(0.0, 1.0);
            
            // Smooth the SPP over time
            self.speech_presence_prob[i] = self.spp_smoothing_factor * self.speech_presence_prob[i] 
                                          + (1.0 - self.spp_smoothing_factor) * spp_instant;
        }
    }
    
    /// Apply Wiener filter with SPP-based blending
    fn apply_wiener_filter(&mut self, noise_spectrum: &[f32], is_speech: bool) {
        // First calculate SPP for each frequency bin
        self.calculate_speech_presence_probability(noise_spectrum, is_speech);
        
        for i in 0..self.magnitude_buffer.len().min(noise_spectrum.len()) {
            let signal_power = self.magnitude_buffer[i] * self.magnitude_buffer[i];
            let noise_power = noise_spectrum[i] * noise_spectrum[i];
            
            // Wiener gain formula
            let snr = signal_power / (noise_power + DENORMAL_PREVENTION);
            let wiener_gain = snr / (1.0 + snr);
            
            // SPP-based blending: g = p * g_wiener + (1-p) * g_floor
            // Where p is the speech presence probability
            let spp = self.speech_presence_prob[i];
            let floor_gain = self.beta_freq[i]; // Use frequency-dependent floor
            
            // Blend between Wiener gain and spectral floor based on SPP
            self.gain_buffer[i] = spp * wiener_gain + (1.0 - spp) * floor_gain;
            
            // Ensure gain is within bounds
            self.gain_buffer[i] = self.gain_buffer[i].clamp(floor_gain, 1.0);
        }
    }
    
    /// Calculate Speech Presence Probability (SPP) for each frequency bin
    fn calculate_speech_presence_probability(&mut self, noise_spectrum: &[f32], global_is_speech: bool) {
        for i in 0..self.magnitude_buffer.len().min(noise_spectrum.len()) {
            // Use a priori SNR for SPP estimation
            let xi = self.a_priori_snr[i];
            
            // SPP estimation using sigmoid function based on a priori SNR
            // Maps SNR to probability: low SNR -> 0, high SNR -> 1
            let spp_instant = if xi > 0.0 {
                // Sigmoid-like function: 1 / (1 + exp(-k*(xi_db - threshold)))
                let xi_db = 10.0 * xi.log10();
                let threshold_db = 3.0; // SNR threshold for 50% speech probability
                let k = 0.3; // Steepness factor
                1.0 / (1.0 + ((-k * (xi_db - threshold_db)).exp()))
            } else {
                0.0
            };
            
            // Apply global VAD decision as a factor
            let vad_factor = if global_is_speech { 1.2 } else { 0.8 };
            let spp_instant = (spp_instant * vad_factor).clamp(0.0, 1.0);
            
            // Smooth the SPP over time to reduce fluctuations
            self.speech_presence_prob[i] = self.spp_smoothing_factor * self.speech_presence_prob[i] 
                                          + (1.0 - self.spp_smoothing_factor) * spp_instant;
        }
    }
    
    /// Apply 2-D gain smoothing (temporal + spectral) to reduce musical noise
    fn apply_gain_smoothing(&mut self) {
        // First apply advanced spectral smoothing (median filter reduces musical noise)
        self.spectral_smoother.smooth(&mut self.gain_buffer);
        
        // Then apply temporal smoothing
        for i in 0..self.gain_buffer.len().min(self.gain_smoothers.len()) {
            self.gain_smoothers[i].set_target(self.gain_buffer[i]);
            self.gain_buffer[i] = self.gain_smoothers[i].next();
        }
        
        // Optional: Apply additional 3-tap smoothing for extra smoothness
        // Disabled by default since spectral_smoother already handles this better
        /*
        if self.gain_buffer.len() >= 3 {
            // Use scratch buffer instead of cloning
            self.smoothed_gains_buf.copy_from_slice(&self.gain_buffer);
            
            // Apply 3-tap smoothing for interior bins
            for i in 1..self.gain_buffer.len() - 1 {
                self.smoothed_gains_buf[i] = 0.25 * self.gain_buffer[i - 1] 
                                            + 0.5 * self.gain_buffer[i] 
                                            + 0.25 * self.gain_buffer[i + 1];
            }
            
            // Handle boundaries with 2-tap smoothing
            if self.gain_buffer.len() > 1 {
                self.smoothed_gains_buf[0] = 0.75 * self.gain_buffer[0] + 0.25 * self.gain_buffer[1];
                let last = self.gain_buffer.len() - 1;
                self.smoothed_gains_buf[last] = 0.25 * self.gain_buffer[last - 1] + 0.75 * self.gain_buffer[last];
            }
            
            // Copy smoothed values back to gain buffer
            self.gain_buffer.copy_from_slice(&self.smoothed_gains_buf);
        }
        */
    }
    
    /// Apply comfort noise to masked regions
    fn apply_comfort_noise(&mut self) {
        if let Some(ref mut gen) = self.comfort_noise_gen {
            // Use scratch buffer instead of allocating
            self.noise_spectrum_buf.copy_from_slice(self.noise_tracker.profile().spectrum());
            gen.set_target_profile(&self.noise_spectrum_buf);
            
            // Use pre-allocated buffers
            gen.generate_into(&mut self.comfort_mag_buf, &mut self.comfort_phase_buf);
            
            // Mix comfort noise with processed signal
            for i in 0..self.magnitude_buffer.len() {
                // Fix: Clamp mix factor to [0, 1] to prevent negative values if gain > 1
                let mix_factor = (1.0 - self.gain_buffer[i]).clamp(0.0, 1.0); // More comfort noise where gain is low
                self.magnitude_buffer[i] = self.magnitude_buffer[i] * (1.0 - mix_factor) 
                                         + self.comfort_mag_buf[i] * mix_factor;
            }
        }
    }
    
    /// Update processing statistics
    fn update_statistics(&mut self, is_speech: bool) {
        self.stats.frames_processed += 1;
        
        if is_speech {
            self.stats.speech_frames += 1;
        } else {
            self.stats.noise_frames += 1;
        }
        
        // Calculate average gain reduction
        let avg_gain = self.gain_buffer.iter().sum::<f32>() / self.gain_buffer.len() as f32;
        let gain_reduction_db = -linear_to_db(avg_gain.max(DENORMAL_PREVENTION));
        
        // Update running average
        let alpha = 0.95;
        self.stats.avg_noise_reduction_db = alpha * self.stats.avg_noise_reduction_db 
                                           + (1.0 - alpha) * gain_reduction_db;
        
        self.stats.peak_noise_reduction_db = self.stats.peak_noise_reduction_db.max(gain_reduction_db);
    }
    
    // ========================================================================
    // Public Processing Methods
    // ========================================================================
    
    /// Process a frame of audio samples
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, ProcessingError> {
        if !self.initialized {
            return Err(ProcessingError::NotInitialized);
        }

        // No longer require input and output to be same size
        // The OLA processor handles its own buffering

        // Extract all needed data before the closure
        let num_bins = self.magnitude_buffer.len();
        // Copy noise spectrum to scratch buffer (avoid allocation)
        self.noise_spectrum_buf.copy_from_slice(self.noise_tracker.profile().spectrum());
        let config = self.config.clone();
        let noise_profile_frozen = self.noise_profile_frozen;
        
        // Extract frequency-dependent parameters
        let alpha_freq = self.alpha_freq.clone();
        let beta_freq = self.beta_freq.clone();
        
        // Extract SPP parameters
        let spp_smoothing_factor = self.spp_smoothing_factor;

        // Extract mutable references we'll need
        let magnitude_buffer = &mut self.magnitude_buffer;
        let phase_buffer = &mut self.phase_buffer;
        let gain_buffer = &mut self.gain_buffer;
        let noise_tracker = &mut self.noise_tracker;
        let gain_smoothers = &mut self.gain_smoothers;
        let comfort_noise_gen = &mut self.comfort_noise_gen;
        let stats = &mut self.stats;
        let a_priori_snr = &mut self.a_priori_snr;
        let prev_output_power = &mut self.prev_output_power;
        let speech_presence_prob = &mut self.speech_presence_prob;
        let noise_spectrum_buf = &self.noise_spectrum_buf;
        let comfort_mag_buf = &mut self.comfort_mag_buf;
        let comfort_phase_buf = &mut self.comfort_phase_buf;
        let smoothed_gains_buf = &mut self.smoothed_gains_buf;

        // Process through FFT processor with all logic inline
        let samples_written = self.fft_processor.process_with_spectrum(input, output, |spectrum| {
            // Extract magnitude only (no phase needed with direct complex multiplication)
            for i in 0..num_bins.min(spectrum.len()) {
                magnitude_buffer[i] = spectrum[i].norm();
            }

            // Update noise profile if adaptive mode
            let is_speech = if config.adaptive_mode && !noise_profile_frozen {
                noise_tracker.process(&magnitude_buffer)
            } else {
                noise_tracker.profile().is_speech(&magnitude_buffer, Some(config.vad_threshold_db))
            };

            // Apply spectral subtraction
            if config.wiener_filter_mode {
                // Wiener filter with SPP-based blending
                
                // First calculate SPP for each frequency bin
                for i in 0..magnitude_buffer.len().min(noise_spectrum_buf.len()) {
                    // Use a priori SNR for SPP estimation
                    let xi = a_priori_snr[i];
                    
                    // SPP estimation using sigmoid function
                    let spp_instant = if xi > 0.0 {
                        let xi_db = 10.0 * xi.log10();
                        let threshold_db = 3.0;
                        let k = 0.3;
                        1.0 / (1.0 + ((-k * (xi_db - threshold_db)).exp()))
                    } else {
                        0.0
                    };
                    
                    // Apply global VAD decision as a factor
                    let vad_factor = if is_speech { 1.2 } else { 0.8 };
                    let spp_instant = (spp_instant * vad_factor).clamp(0.0, 1.0);
                    
                    // Smooth the SPP over time
                    speech_presence_prob[i] = spp_smoothing_factor * speech_presence_prob[i] 
                                            + (1.0 - spp_smoothing_factor) * spp_instant;
                }
                
                // Apply Wiener filter with SPP blending
                for i in 0..magnitude_buffer.len().min(noise_spectrum_buf.len()) {
                    let signal_power = magnitude_buffer[i] * magnitude_buffer[i];
                    let noise_power = noise_spectrum_buf[i] * noise_spectrum_buf[i];

                    let snr = signal_power / (noise_power + DENORMAL_PREVENTION);
                    let wiener_gain = snr / (1.0 + snr);
                    
                    // SPP-based blending: g = p * g_wiener + (1-p) * g_floor
                    let spp = speech_presence_prob[i];
                    let floor_gain = beta_freq[i];
                    
                    gain_buffer[i] = spp * wiener_gain + (1.0 - spp) * floor_gain;
                    gain_buffer[i] = gain_buffer[i].clamp(floor_gain, 1.0);
                }
            } else {
                // Basic spectral subtraction with frequency-dependent parameters
                let exponent = config.subtraction_exponent;
                const ALPHA_XI: f32 = 0.98; // Decision-directed smoothing factor

                for i in 0..magnitude_buffer.len().min(noise_spectrum_buf.len()) {
                    let signal_power = magnitude_buffer[i].powf(exponent);
                    let noise_power = noise_spectrum_buf[i].powf(exponent);
                    
                    // Decision-directed a priori SNR estimation
                    let posterior_snr = (signal_power / (noise_power + DENORMAL_PREVENTION)).max(0.0);
                    
                    // First term: smoothed gain from previous frame
                    let prev_gain_term = if prev_output_power[i] > 0.0 {
                        ALPHA_XI * (prev_output_power[i] / (noise_power + DENORMAL_PREVENTION))
                    } else {
                        0.0
                    };
                    
                    // Second term: ML estimate from current frame
                    let ml_term = (1.0 - ALPHA_XI) * (posterior_snr - 1.0).max(0.0);
                    
                    // Combined a priori SNR
                    a_priori_snr[i] = prev_gain_term + ml_term;
                    
                    // Adaptive alpha based on a priori SNR
                    let mut alpha = if a_priori_snr[i] > 10.0 {
                        alpha_freq[i] * 0.7  // Less aggressive when SNR is high
                    } else if a_priori_snr[i] < 1.0 {
                        alpha_freq[i] * 1.3  // More aggressive when SNR is low
                    } else {
                        alpha_freq[i]
                    };
                    
                    // Further adjust alpha based on speech detection
                    if !is_speech {
                        alpha *= 1.2;
                    }
                    
                    // Use frequency-dependent beta
                    let beta = beta_freq[i];

                    let subtracted_power = signal_power - alpha * noise_power;
                    let floored_power = (beta * noise_power).max(0.0);

                    let output_power = if subtracted_power > floored_power {
                        subtracted_power
                    } else {
                        floored_power
                    };
                    
                    // Store output power for next frame's a priori SNR estimation
                    prev_output_power[i] = output_power;

                    // Calculate gain
                    let ratio = (output_power / (signal_power + DENORMAL_PREVENTION)).max(0.0);
                    let mut g = ratio.powf(1.0 / exponent);
                    
                    // Clamp to [floor, 1.0]
                    g = g.min(1.0);
                    
                    // Use frequency-dependent gain floor
                    let gain_floor = beta.min(1.0);
                    g = g.max(gain_floor);
                    
                    gain_buffer[i] = g;
                    // Don't modify magnitude_buffer - we'll apply gain directly to complex spectrum
                }
            }

            // Apply comfort noise if enabled (needs phase extraction for noise generation)
            if config.enable_comfort_noise && !is_speech {
                if let Some(ref mut gen) = comfort_noise_gen {
                    // Extract phase only when needed for comfort noise
                    for i in 0..num_bins.min(spectrum.len()) {
                        phase_buffer[i] = spectrum[i].arg();
                    }
                    
                    gen.set_target_profile(&noise_spectrum_buf);

                    // Use scratch buffers instead of allocating new vectors
                    gen.generate_into(comfort_mag_buf, comfort_phase_buf);

                    // Mix comfort noise with original signal at magnitude level
                    for i in 0..magnitude_buffer.len() {
                        // Fix: Clamp mix factor to [0, 1] to prevent negative values if gain > 1
                        let mix_factor = (1.0 - gain_buffer[i]).clamp(0.0, 1.0);
                        let mixed_magnitude = magnitude_buffer[i] * (1.0 - mix_factor) 
                            + comfort_mag_buf[i] * mix_factor;
                        // Update gain to reflect the mixed magnitude
                        // Clamp to maintain invariant that gain ∈ [floor, 1.0]
                        let floor = beta_freq[i].min(1.0);
                        gain_buffer[i] = (mixed_magnitude / (magnitude_buffer[i] + DENORMAL_PREVENTION))
                            .min(1.0)
                            .max(floor);
                    }
                }
            }

            // Apply 2-D gain smoothing if enabled (temporal + spectral)
            if config.enable_smoothing {
                // First apply temporal smoothing
                for i in 0..gain_buffer.len().min(gain_smoothers.len()) {
                    gain_smoothers[i].set_target(gain_buffer[i]);
                    gain_buffer[i] = gain_smoothers[i].next();
                }
                
                // Then apply spectral smoothing (3-tap symmetric filter)
                if gain_buffer.len() >= 3 {
                    // Use scratch buffer instead of allocating
                    smoothed_gains_buf.copy_from_slice(gain_buffer);
                    
                    // Apply 3-tap smoothing for interior bins
                    for i in 1..gain_buffer.len() - 1 {
                        smoothed_gains_buf[i] = 0.25 * gain_buffer[i - 1] 
                                              + 0.5 * gain_buffer[i] 
                                              + 0.25 * gain_buffer[i + 1];
                    }
                    
                    // Handle boundaries with 2-tap smoothing
                    if gain_buffer.len() > 1 {
                        smoothed_gains_buf[0] = 0.75 * gain_buffer[0] + 0.25 * gain_buffer[1];
                        let last = gain_buffer.len() - 1;
                        smoothed_gains_buf[last] = 0.25 * gain_buffer[last - 1] + 0.75 * gain_buffer[last];
                    }
                    
                    // Copy smoothed values back to gain buffer
                    gain_buffer.copy_from_slice(smoothed_gains_buf);
                }
            }

            // Apply gains directly to complex spectrum (avoiding from_polar)
            for i in 0..num_bins.min(spectrum.len()) {
                // Multiply complex number by real gain - preserves phase automatically
                spectrum[i] *= gain_buffer[i];
            }

            // Maintain conjugate symmetry
            if spectrum.len() == config.fft_size {
                for i in 1..num_bins - 1 {
                    if i < spectrum.len() && config.fft_size - i < spectrum.len() {
                        spectrum[config.fft_size - i] = spectrum[i].conj();
                    }
                }
                // Zero imaginary parts of DC and Nyquist for numerical hygiene
                spectrum[0].im = 0.0;
                if num_bins > 1 && num_bins - 1 < spectrum.len() {
                    spectrum[num_bins - 1].im = 0.0;
                }
            }

            // Update statistics
            stats.frames_processed += 1;
            if is_speech {
                stats.speech_frames += 1;
            } else {
                stats.noise_frames += 1;
            }

            let avg_gain = gain_buffer.iter().sum::<f32>() / gain_buffer.len() as f32;
            let gain_reduction_db = -linear_to_db(avg_gain.max(DENORMAL_PREVENTION));

            let alpha = 0.95;
            stats.avg_noise_reduction_db = alpha * stats.avg_noise_reduction_db 
                + (1.0 - alpha) * gain_reduction_db;
            stats.peak_noise_reduction_db = stats.peak_noise_reduction_db.max(gain_reduction_db);
        })?;

        // Return the number of samples actually written to output
        Ok(samples_written)
    }    
    /// Process a single sample (accumulates internally for FFT frames)
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        if !self.initialized {
            return sample;
        }
        
        // Add to input buffer
        if self.sample_buffer_index < self.sample_buffer.len() {
            self.sample_buffer[self.sample_buffer_index] = sample;
            self.sample_buffer_index += 1;
        }
        
        // Process when we have enough samples
        if self.sample_buffer_index >= self.sample_buffer.len() {
            // Use a local buffer to avoid borrow conflicts
            let mut temp_output = vec![0.0; self.sample_buffer.len()];
            
            // Clone input to avoid borrow conflicts
            let input_copy = self.sample_buffer.clone();
            
            // Process the frame
            if self.process_frame(&input_copy, &mut temp_output).is_ok() {
                // Store output
                for i in 0..temp_output.len().min(self.output_buffer.len() - self.config.fft_size) {
                    self.output_buffer[i + self.config.fft_size] = temp_output[i];
                }
            }
            
            // Reset input buffer
            self.sample_buffer_index = 0;
            self.sample_buffer.fill(0.0);
        }
        
        // Return output sample
        let output = if self.output_buffer_index < self.output_buffer.len() {
            let out = self.output_buffer[self.output_buffer_index];
            self.output_buffer_index += 1;
            
            // Shift buffer when half consumed
            if self.output_buffer_index >= self.config.fft_size {
                self.output_buffer.rotate_left(self.config.fft_size);
                self.output_buffer_index -= self.config.fft_size;
            }
            
            out
        } else {
            0.0
        };
        
        output
    }
    
    /// Get the hop size used by the FFT processor
    pub fn hop_size(&self) -> usize {
        self.fft_processor.hop_size()
    }
    
    /// Update noise profile during silence
    pub fn update_noise_profile(&mut self, input: &[f32]) -> Result<(), ProcessingError> {
        // Process input in FFT frames
        let fft_size = self.config.fft_size;
        
        for chunk in input.chunks(fft_size) {
            if chunk.len() == fft_size {
                let mut padded = chunk.to_vec();
                
                // Apply window
                let window = match self.config.window_type {
                    WindowType::Hann => crate::utils::hann_window(fft_size),
                    WindowType::Hamming => crate::utils::hamming_window(fft_size),
                    _ => vec![1.0; fft_size],
                };
                
                for (s, w) in padded.iter_mut().zip(window.iter()) {
                    *s *= w;
                }
                
                // Convert to frequency domain
                let mut complex: Vec<Complex32> = padded.iter()
                    .map(|&s| Complex32::new(s, 0.0))
                    .collect();
                
                // Would need actual FFT here - simplified for now
                // In real implementation, use rustfft
                
                // Extract magnitude spectrum
                let magnitudes: Vec<f32> = complex.iter()
                    .take(fft_size / 2 + 1)
                    .map(|c| c.norm())
                    .collect();
                
                // Update noise profile
                self.noise_tracker.profile_mut().update(&magnitudes)?;
            }
        }
        
        Ok(())
    }
    
    /// Process a buffer of samples
    pub fn process_buffer(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ProcessingError> {
        if input.len() != output.len() {
            return Err(ProcessingError::BufferSizeMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }
        
        // Clear output buffer first
        output.fill(0.0);
        
        // Process input through the FFT processor which uses overlap-add
        // The amount of output may not match input due to buffering and latency
        let mut input_offset = 0;
        let mut output_offset = 0;
        
        // Use hop-sized chunks for predictable behavior
        let hop_size = self.fft_processor.hop_size();
        
        while input_offset < input.len() && output_offset < output.len() {
            // Feed input in hop-sized chunks (or remaining input)
            let chunk_size = (input.len() - input_offset).min(hop_size);
            let in_chunk = &input[input_offset..input_offset + chunk_size];
            
            // Process chunk and get actual samples written
            let out_remaining = &mut output[output_offset..];
            let samples_written = self.process_frame(in_chunk, out_remaining)?;
            
            input_offset += chunk_size;
            output_offset += samples_written;
        }
        
        Ok(())
    }
    
    /// Process buffer in-place
    pub fn process_buffer_in_place(&mut self, buffer: &mut [f32]) {
        let temp_input = buffer.to_vec();
        let _ = self.process_buffer(&temp_input, buffer);
    }
    
    // ========================================================================
    // Configuration Methods
    // ========================================================================
    
    /// Enable or disable adaptive noise estimation
    pub fn set_noise_estimation_enabled(&mut self, enabled: bool) {
        self.config.adaptive_mode = enabled;
    }
    
    // ========================================================================
    // API Ergonomics - Parameter Setters
    // ========================================================================
    
    /// Set the over-subtraction factor (alpha)
    /// Typical range: 1.0 - 3.0
    pub fn set_alpha(&mut self, alpha: f32) {
        self.config.alpha = alpha.max(0.5).min(5.0);
        // Update frequency-dependent alpha
        self.update_frequency_dependent_params();
    }
    
    /// Set the spectral floor (beta) 
    /// Typical range: 0.001 - 0.1
    pub fn set_beta(&mut self, beta: f32) {
        self.config.beta = beta.max(0.0001).min(0.5);
        // Update frequency-dependent beta
        self.update_frequency_dependent_params();
    }
    
    /// Set the subtraction exponent
    /// Typical values: 1.0 (magnitude), 2.0 (power)
    pub fn set_subtraction_exponent(&mut self, exponent: f32) {
        self.config.subtraction_exponent = exponent.max(0.5).min(4.0);
    }
    
    /// Set comfort noise gain
    /// Range: 0.0 (disabled) - 0.1 (moderate)
    pub fn set_comfort_noise_gain(&mut self, gain: f32) {
        self.config.comfort_noise_gain = gain.max(0.0).min(0.1);
        self.config.enable_comfort_noise = gain > 0.0;  // Enable when gain > 0
        if let Some(ref mut gen) = self.comfort_noise_gen {
            gen.set_gain(gain);
        }
    }
    
    /// Enable/disable comfort noise
    pub fn set_comfort_noise_enabled(&mut self, enabled: bool) {
        self.config.enable_comfort_noise = enabled;
        if enabled && self.comfort_noise_gen.is_none() {
            let mut gen = ComfortNoiseGenerator::new(self.magnitude_buffer.len());
            gen.set_gain(self.config.comfort_noise_gain);
            self.comfort_noise_gen = Some(gen);
        }
    }
    
    /// Enable/disable Wiener filter mode
    pub fn set_wiener_filter_mode(&mut self, enabled: bool) {
        self.config.wiener_filter_mode = enabled;
    }
    
    /// Check if Wiener filter mode is enabled
    pub fn is_wiener_mode(&self) -> bool {
        self.config.wiener_filter_mode
    }
    
    /// Enable/disable adaptive mode
    pub fn set_adaptive_mode(&mut self, enabled: bool) {
        self.config.adaptive_mode = enabled;
    }
    
    /// Set VAD threshold in dB
    pub fn set_vad_threshold(&mut self, threshold_db: f32) {
        self.config.vad_threshold_db = threshold_db.max(-60.0).min(0.0);
    }
    
    /// Freeze the noise profile (prevent updates)
    pub fn freeze_noise_profile(&mut self) {
        self.noise_profile_frozen = true;
    }
    
    /// Unfreeze the noise profile (allow updates)
    pub fn unfreeze_noise_profile(&mut self) {
        self.noise_profile_frozen = false;
    }
    
    /// Check if noise profile is frozen
    pub fn is_noise_profile_frozen(&self) -> bool {
        self.noise_profile_frozen
    }
    
    // ========================================================================
    // API Ergonomics - Parameter Getters
    // ========================================================================
    
    /// Get current alpha value
    pub fn alpha(&self) -> f32 {
        self.config.alpha
    }
    
    /// Get current beta value
    pub fn beta(&self) -> f32 {
        self.config.beta
    }
    
    /// Get current subtraction exponent
    pub fn subtraction_exponent(&self) -> f32 {
        self.config.subtraction_exponent
    }
    
    /// Get current comfort noise gain
    pub fn comfort_noise_gain(&self) -> f32 {
        self.config.comfort_noise_gain
    }
    
    /// Check if comfort noise is enabled
    pub fn is_comfort_noise_enabled(&self) -> bool {
        self.config.enable_comfort_noise
    }
    
    /// Check if in Wiener filter mode
    pub fn is_wiener_filter_mode(&self) -> bool {
        self.config.wiener_filter_mode
    }
    
    /// Check if in adaptive mode
    pub fn is_adaptive_mode(&self) -> bool {
        self.config.adaptive_mode
    }
    
    /// Get VAD threshold
    pub fn vad_threshold(&self) -> f32 {
        self.config.vad_threshold_db
    }
    
    /// Get a copy of the current configuration
    pub fn config(&self) -> SpectralSubtractionConfig {
        self.config.clone()
    }
    
    /// Update frequency-dependent parameters after config changes
    fn update_frequency_dependent_params(&mut self) {
        let num_bins = self.magnitude_buffer.len();
        let freq_per_bin = self.config.sample_rate / self.config.fft_size as f32;
        
        // Adapt speech band based on sample rate
        let speech_low = if self.config.sample_rate >= 44100.0 { 200.0 } else { 300.0 };
        let speech_high = if self.config.sample_rate >= 44100.0 { 7000.0 } else { 3400.0 };
        
        for i in 0..num_bins {
            let freq_hz = i as f32 * freq_per_bin;
            
            // Alpha: Less aggressive in speech formant regions
            if freq_hz >= speech_low && freq_hz <= speech_high {
                self.alpha_freq[i] = self.config.alpha;
            } else {
                self.alpha_freq[i] = self.config.alpha * 1.5;
            }
            
            // Beta: Smaller floor in speech regions
            if freq_hz >= speech_low && freq_hz <= speech_high {
                self.beta_freq[i] = self.config.beta * 0.5;
            } else if freq_hz < 100.0 || freq_hz > 8000.0 {
                self.beta_freq[i] = self.config.beta * 2.0;
            } else {
                self.beta_freq[i] = self.config.beta;
            }
        }
    }
    
    // ========================================================================
    // Noise Management Methods
    // ========================================================================
    
    /// Learn noise profile from noise-only samples
    pub fn learn_noise_profile(&mut self, samples: &[f32]) -> Result<(), ProcessingError> {
        if samples.is_empty() {
            return Err(ProcessingError::InvalidConfiguration(
                "No samples provided for noise learning".to_string()
            ));
        }
        
        // Temporarily disable adaptive mode
        let prev_adaptive = self.config.adaptive_mode;
        self.config.adaptive_mode = false;
        
        // Process samples to learn noise
        self.update_noise_profile(samples)?;
        
        // Update comfort noise generator if enabled
        if let Some(ref mut gen) = self.comfort_noise_gen {
            let noise_spectrum = self.noise_tracker.profile().spectrum().to_vec();
            gen.set_target_profile(&noise_spectrum);
        }
        
        // Restore adaptive mode
        self.config.adaptive_mode = prev_adaptive;
        
        Ok(())
    }
    
    /// Reset noise profile to default
    pub fn reset_noise_profile(&mut self) {
        self.noise_tracker.reset();
        self.noise_profile_frozen = false;
    }
    
    // ========================================================================
    // State Management Methods
    // ========================================================================
    
    /// Reset all processing state
    pub fn reset(&mut self) {
        self.fft_processor.reset();
        self.noise_tracker.reset();
        
        self.magnitude_buffer.fill(0.0);
        self.phase_buffer.fill(0.0);
        self.gain_buffer.fill(1.0);
        
        for smoother in &mut self.gain_smoothers {
            smoother.reset(1.0);
        }
        
        self.sample_buffer.fill(0.0);
        self.sample_buffer_index = 0;
        self.output_buffer.fill(0.0);
        self.output_buffer_index = 0;
        
        // Reset DSP improvement fields
        self.a_priori_snr.fill(1.0);
        self.prev_output_power.fill(0.0);
        self.speech_presence_prob.fill(0.0);
        
        self.stats = SubtractionStats::default();
    }
    
    /// Get processing latency in samples
    pub fn latency_samples(&self) -> usize {
        self.fft_processor.latency_samples()
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> SubtractionStats {
        self.stats.clone()
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for SpectralSubtractor
pub struct SpectralSubtractorBuilder {
    config: SpectralSubtractionConfig,
}

impl SpectralSubtractorBuilder {
    /// Create new builder with default config
    pub fn new() -> Self {
        Self {
            config: SpectralSubtractionConfig::default(),
        }
    }
    
    /// Set sample rate
    pub fn sample_rate(mut self, rate: f32) -> Self {
        self.config.sample_rate = rate;
        self
    }
    
    /// Set FFT size
    pub fn fft_size(mut self, size: usize) -> Self {
        self.config.fft_size = size;
        self
    }
    
    /// Set over-subtraction factor
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.config.alpha = alpha;
        self
    }
    
    /// Set spectral floor
    pub fn beta(mut self, beta: f32) -> Self {
        self.config.beta = beta;
        self
    }
    
    /// Set window type
    pub fn window_type(mut self, window: WindowType) -> Self {
        self.config.window_type = window;
        self
    }
    
    /// Set overlap percentage
    pub fn overlap(mut self, percent: f32) -> Self {
        self.config.overlap_percent = percent;
        self
    }
    
    /// Enable Wiener filter mode
    pub fn wiener_mode(mut self, enabled: bool) -> Self {
        self.config.wiener_filter_mode = enabled;
        self
    }
    
    /// Enable comfort noise
    pub fn comfort_noise(mut self, enabled: bool, gain: f32) -> Self {
        self.config.enable_comfort_noise = enabled;
        self.config.comfort_noise_gain = gain;
        self
    }
    
    /// Build the spectral subtractor
    pub fn build(self) -> Result<SpectralSubtractor, ProcessingError> {
        SpectralSubtractor::new(self.config)
    }
}

impl Default for SpectralSubtractorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_validation() {
        let mut config = SpectralSubtractionConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid alpha
        config.alpha = -1.0;
        assert!(config.validate().is_err());
        config.alpha = 2.0;
        assert!(config.validate().is_ok());
        
        // Test invalid beta
        config.beta = 0.0;
        assert!(config.validate().is_err());
        config.beta = 0.1;
        assert!(config.validate().is_ok());
        
        // Test invalid FFT size
        config.fft_size = 100;
        assert!(config.validate().is_err());
        config.fft_size = 512;
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_spectral_subtractor_creation() {
        let subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        assert_eq!(subtractor.config.sample_rate, 48000.0);
        assert!(subtractor.initialized);
    }
    
    #[test]
    fn test_builder_pattern() {
        let result = SpectralSubtractorBuilder::new()
            .sample_rate(44100.0)
            .fft_size(1024)
            .alpha(2.5)
            .beta(0.05)
            .wiener_mode(true)
            .build();
        
        assert!(result.is_ok());
        let subtractor = result.unwrap();
        assert_eq!(subtractor.config.sample_rate, 44100.0);
        assert_eq!(subtractor.config.fft_size, 1024);
        assert_eq!(subtractor.config.alpha, 2.5);
        assert_eq!(subtractor.config.beta, 0.05);
        assert!(subtractor.config.wiener_filter_mode);
    }
    
    #[test]
    fn test_process_empty_buffer() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        let input = vec![];
        let mut output = vec![];
        
        let result = subtractor.process_buffer(&input, &mut output);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_process_buffer_mismatch() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        let input = vec![0.0; 100];
        let mut output = vec![0.0; 50];
        
        let result = subtractor.process_buffer(&input, &mut output);
        assert!(matches!(result, Err(ProcessingError::BufferSizeMismatch { .. })));
    }
    
    #[test]
    fn test_single_sample_processing() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        
        // Process some samples
        for i in 0..1000 {
            let input = (i as f32 * 0.01).sin();
            let _ = subtractor.process_sample(input);
        }
        
        // Should have processed some frames
        assert!(subtractor.stats.frames_processed > 0);
    }
    
    #[test]
    fn test_noise_profile_management() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        
        // Generate noise samples
        let noise = vec![0.01; 1024];
        
        // Learn noise profile
        assert!(subtractor.learn_noise_profile(&noise).is_ok());
        
        // Test freeze/unfreeze
        subtractor.freeze_noise_profile();
        assert!(subtractor.noise_profile_frozen);
        
        subtractor.unfreeze_noise_profile();
        assert!(!subtractor.noise_profile_frozen);
        
        // Reset
        subtractor.reset_noise_profile();
    }
    
    #[test]
    fn test_parameter_updates() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        
        subtractor.set_alpha(3.0);
        assert_eq!(subtractor.config.alpha, 3.0);
        
        subtractor.set_alpha(10.0); // Should clamp
        assert_eq!(subtractor.config.alpha, MAX_ALPHA);
        
        subtractor.set_beta(0.001);
        assert!(subtractor.config.beta >= MIN_BETA);
        
        subtractor.set_noise_estimation_enabled(false);
        assert!(!subtractor.config.adaptive_mode);
    }
    
    #[test]
    fn test_latency_calculation() {
        let subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        let latency = subtractor.latency_samples();
        
        // Latency should be related to FFT size
        assert!(latency > 0);
    }
    
    #[test]
    fn test_statistics_tracking() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        
        // Process some frames
        let input = vec![0.0; 512];
        let mut output = vec![0.0; 512];
        
        for _ in 0..10 {
            let _ = subtractor.process_frame(&input, &mut output);
        }
        
        let stats = subtractor.get_stats();
        assert!(stats.frames_processed > 0);
    }
    
    #[test]
    fn test_reset_functionality() {
        let mut subtractor = SpectralSubtractor::with_sample_rate(48000.0);
        
        // Process some data
        let input = vec![0.1; 512];
        let mut output = vec![0.0; 512];
        let _ = subtractor.process_frame(&input, &mut output);
        
        // Reset
        subtractor.reset();
        
        // Check state is cleared
        assert_eq!(subtractor.stats.frames_processed, 0);
        assert_eq!(subtractor.sample_buffer_index, 0);
        assert!(subtractor.gain_buffer.iter().all(|&g| g == 1.0));
    }
    
    // ========================================================================
    // Tests from Review Suggestions
    // ========================================================================
    
    /// Test 1: No explosion - white noise should not explode
    #[test]
    fn test_no_explosion_white_noise() {
        use rand::{SeedableRng, Rng};
        use rand::rngs::SmallRng;
        
        let sample_rate = 48000.0;
        let test_duration_samples = 4800; // 100ms
        
        // Test with various alpha/beta combinations
        let alpha_values = [0.5, 1.0, 2.0, 3.0, 4.0];
        let beta_values = [0.001, 0.01, 0.02, 0.05, 0.1];
        
        for &alpha in &alpha_values {
            for &beta in &beta_values {
                let mut subtractor = SpectralSubtractorBuilder::new()
                    .sample_rate(sample_rate)
                    .alpha(alpha)
                    .beta(beta)
                    .comfort_noise(false, 0.0) // Disable comfort noise for pure test
                    .build()
                    .expect("Failed to build subtractor");
                
                // Generate white noise
                let mut rng = SmallRng::seed_from_u64(42);
                let input: Vec<f32> = (0..test_duration_samples)
                    .map(|_| rng.gen_range(-0.5..0.5))
                    .collect();
                
                // Calculate input RMS
                let input_rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
                
                // Process the noise
                let mut output = vec![0.0; test_duration_samples];
                subtractor.process_buffer(&input, &mut output)
                    .expect("Processing failed");
                
                // Calculate output RMS
                let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
                
                // Assert output doesn't explode (allow 20% increase due to processing)
                assert!(
                    output_rms <= input_rms * 1.2,
                    "Output RMS ({}) exceeded input RMS ({}) by more than 20% with alpha={}, beta={}",
                    output_rms, input_rms, alpha, beta
                );
                
                // Also check no individual sample explodes
                let max_output = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let max_input = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                assert!(
                    max_output <= max_input * 1.5,
                    "Max output ({}) exceeded max input ({}) by more than 50% with alpha={}, beta={}",
                    max_output, max_input, alpha, beta
                );
            }
        }
    }
    
    /// Test 2: Silence stability
    #[test]
    fn test_silence_stability() {
        let sample_rate = 48000.0;
        let test_duration_samples = 4800; // 100ms
        
        // Test without comfort noise
        {
            let mut subtractor = SpectralSubtractorBuilder::new()
                .sample_rate(sample_rate)
                .comfort_noise(false, 0.0)
                .build()
                .expect("Failed to build subtractor");
            
            // Process silence
            let input = vec![0.0; test_duration_samples];
            let mut output = vec![0.0; test_duration_samples];
            
            for _ in 0..10 { // Process multiple times to ensure stability
                subtractor.process_buffer(&input, &mut output)
                    .expect("Processing failed");
                
                // Output should remain near zero
                let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
                assert!(
                    output_rms < 1e-6,
                    "Output RMS ({}) not near zero for silence input without comfort noise",
                    output_rms
                );
            }
        }
        
        // Test with comfort noise
        {
            let mut subtractor = SpectralSubtractorBuilder::new()
                .sample_rate(sample_rate)
                .comfort_noise(true, 0.01)
                .build()
                .expect("Failed to build subtractor");
            
            // Process silence multiple times
            let input = vec![0.0; test_duration_samples];
            let mut output = vec![0.0; test_duration_samples];
            let mut rms_values = Vec::new();
            
            for _ in 0..10 {
                subtractor.process_buffer(&input, &mut output)
                    .expect("Processing failed");
                
                let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
                rms_values.push(output_rms);
            }
            
            // Check RMS is stable and bounded
            let mean_rms = rms_values.iter().sum::<f32>() / rms_values.len() as f32;
            let max_deviation = rms_values.iter()
                .map(|&rms| (rms - mean_rms).abs())
                .fold(0.0f32, f32::max);
            
            assert!(
                mean_rms < 0.1,
                "Comfort noise RMS ({}) too high",
                mean_rms
            );
            
            // Only check stability if there's actual comfort noise
            if mean_rms > 0.001 {
                assert!(
                    max_deviation / mean_rms < 0.2,
                    "Comfort noise not stable: max deviation {} from mean {}",
                    max_deviation, mean_rms
                );
            }
        }
    }
    
    /// Test 3: Energy consistency with pure tone
    #[test]
    fn test_energy_consistency_pure_tone() {
        let sample_rate = 48000.0;
        let fft_size = 512;
        let tone_freq = 1000.0; // 1kHz tone
        let test_duration_samples = 4800;
        
        let mut subtractor = SpectralSubtractorBuilder::new()
            .sample_rate(sample_rate)
            .fft_size(fft_size)
            .comfort_noise(false, 0.0)
            .alpha(2.0)
            .beta(0.02)
            .build()
            .expect("Failed to build subtractor");
        
        // First, learn noise profile from silence
        let noise = vec![0.001; 2048]; // Very quiet noise
        subtractor.learn_noise_profile(&noise)
            .expect("Failed to learn noise");
        
        // Generate pure tone
        let omega = 2.0 * std::f32::consts::PI * tone_freq / sample_rate;
        let input: Vec<f32> = (0..test_duration_samples)
            .map(|i| 0.5 * (i as f32 * omega).sin())
            .collect();
        
        // Process the tone
        let mut output = vec![0.0; test_duration_samples];
        subtractor.process_buffer(&input, &mut output)
            .expect("Processing failed");
        
        // Calculate energy in frequency domain using FFT
        // The tone should be attenuated but proportionally
        let input_energy = input.iter().map(|x| x * x).sum::<f32>();
        let output_energy = output.iter().map(|x| x * x).sum::<f32>();
        
        // The attenuation should be consistent
        let attenuation = output_energy / input_energy;
        
        // Attenuation should be between 0 and 1 (no amplification)
        assert!(
            attenuation > 0.0 && attenuation <= 1.0,
            "Attenuation {} out of expected range [0, 1]",
            attenuation
        );
        
        // For a pure tone with minimal noise, attenuation should be minimal
        // (most energy should be preserved)
        assert!(
            attenuation > 0.5,
            "Too much attenuation ({}) for pure tone",
            attenuation
        );
    }
    
    /// Test 4: Smoothing correctness
    #[test]
    fn test_smoothing_correctness() {
        let sample_rate = 48000.0;
        let fft_size = 512;
        let hop_size = fft_size / 2; // 50% overlap
        let smoothing_ms = 5.0; // 5ms smoothing time
        
        // Create subtractor with smoothing enabled
        let mut subtractor = SpectralSubtractorBuilder::new()
            .sample_rate(sample_rate)
            .fft_size(fft_size)
            // Smoothing is enabled by default, set appropriate parameters
            .alpha(2.0)
            .beta(0.02)
            .build()
            .expect("Failed to build subtractor");
        
        // Expected settling time in frames
        // After hop-aware fix: frames_per_second = sample_rate / hop_size
        let frames_per_second = sample_rate / hop_size as f32;
        let expected_settling_frames = (smoothing_ms * 0.001 * frames_per_second * 3.0) as usize; // 3 time constants for ~95% settling
        
        // Create input that will cause a step change in gain
        // Start with silence, then sudden loud signal
        let mut input = vec![0.0; fft_size * 20]; // Enough for multiple frames
        for i in fft_size*5..fft_size*15 {
            input[i] = 0.5; // Step to constant value
        }
        
        let mut output = vec![0.0; input.len()];
        
        // Process frame by frame to track gain changes
        let mut frame_gains = Vec::new();
        for chunk_idx in 0..(input.len() / hop_size) {
            let start = chunk_idx * hop_size;
            if start + hop_size > input.len() { break; }
            
            let frame_in = &input[start..start + hop_size];
            let mut frame_out = vec![0.0; hop_size];
            
            subtractor.process_frame(frame_in, &mut frame_out)
                .expect("Frame processing failed");
            
            // Estimate gain from output/input ratio
            let in_energy = frame_in.iter().map(|x| x * x).sum::<f32>();
            let out_energy = frame_out.iter().map(|x| x * x).sum::<f32>();
            
            let gain = if in_energy > 1e-10 {
                (out_energy / in_energy).sqrt()
            } else {
                1.0
            };
            
            frame_gains.push(gain);
            
            // Copy output
            for i in 0..hop_size {
                if start + i < output.len() {
                    output[start + i] = frame_out[i];
                }
            }
        }
        
        // Find where the step change occurs
        let step_frame = 5 * fft_size / hop_size;
        
        if frame_gains.len() > step_frame + expected_settling_frames {
            // Check that gain settles within expected time
            let initial_gain = frame_gains[step_frame - 1];
            let final_gain = frame_gains[step_frame + expected_settling_frames];
            let halfway_gain = frame_gains[step_frame + expected_settling_frames / 2];
            
            // The gain should change smoothly
            assert!(
                (halfway_gain - initial_gain).abs() > 0.0 || (final_gain - initial_gain).abs() > 0.0,
                "No gain change detected during step input"
            );
            
            // After settling time, gain should be stable
            if frame_gains.len() > step_frame + expected_settling_frames + 5 {
                let settled_gains = &frame_gains[step_frame + expected_settling_frames..step_frame + expected_settling_frames + 5];
                let mean_settled = settled_gains.iter().sum::<f32>() / settled_gains.len() as f32;
                let max_deviation = settled_gains.iter()
                    .map(|&g| (g - mean_settled).abs())
                    .fold(0.0f32, f32::max);
                
                // Allow 20% deviation due to buffering and processing delays
                assert!(
                    max_deviation < 0.2 * mean_settled.abs() + 0.02,
                    "Gain not settled after {} frames: deviation {} from mean {}",
                    expected_settling_frames, max_deviation, mean_settled
                );
            }
        }
    }
}
