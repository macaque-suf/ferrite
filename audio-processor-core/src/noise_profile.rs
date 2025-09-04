//! Noise profile estimation and management for spectral subtraction
//! 
//! This module provides adaptive noise profiling with statistical analysis
//! for robust noise reduction in varying acoustic environments.
//!
//! # Thread Safety
//! The structures in this module are not thread-safe. For concurrent access,
//! wrap in Arc<Mutex<_>> or Arc<RwLock<_>>.

use crate::utils::{linear_to_db, db_to_linear, calculate_rms, DENORMAL_PREVENTION};
use std::collections::VecDeque;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

// ============================================================================
// Constants
// ============================================================================

/// Minimum frames required for initial noise estimation
pub const MIN_FRAMES_FOR_ESTIMATION: usize = 10;

/// Maximum frames to use for noise estimation (memory limit)
pub const MAX_FRAMES_FOR_ESTIMATION: usize = 100;

/// Default smoothing factor for exponential averaging (0.0-1.0)
pub const DEFAULT_SMOOTHING_FACTOR: f32 = 0.98;

/// Voice Activity Detection (VAD) threshold in dB
pub const DEFAULT_VAD_THRESHOLD_DB: f32 = 6.0;

/// Minimum noise floor in dB (safety limit)
pub const MIN_NOISE_FLOOR_DB: f32 = -80.0;

/// Maximum noise floor in dB (sanity check)
pub const MAX_NOISE_FLOOR_DB: f32 = -20.0;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum NoiseProfileError {
    InsufficientData { frames_needed: usize, frames_available: usize },
    InvalidConfiguration { message: String },
    ProfileLocked,
    BufferSizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for NoiseProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoiseProfileError::InsufficientData { frames_needed, frames_available } => 
                write!(f, "Insufficient data for noise estimation: need {}, have {}", 
                       frames_needed, frames_available),
            NoiseProfileError::InvalidConfiguration { message } => 
                write!(f, "Invalid configuration: {}", message),
            NoiseProfileError::ProfileLocked => 
                write!(f, "Cannot update locked noise profile"),
            NoiseProfileError::BufferSizeMismatch { expected, actual } => 
                write!(f, "Buffer size mismatch: expected {}, got {}", expected, actual),
        }
    }
}

impl std::error::Error for NoiseProfileError {}

// ============================================================================
// Noise Profile Statistics
// ============================================================================

/// Statistical information about the noise profile
#[derive(Debug, Clone)]
pub struct NoiseStatistics {
    /// Mean noise level in dB
    pub mean_db: f32,
    
    /// Standard deviation in dB space
    pub std_dev_db: f32,
    
    /// Minimum noise level in dB
    pub min_db: f32,
    
    /// Maximum noise level in dB
    pub max_db: f32,
    
    /// Median noise level in dB (approximate)
    pub median_db: f32,
    
    /// Spectral centroid (center of mass) in Hz
    pub spectral_centroid: f32,
    
    /// Spectral flatness (0.0 = tonal, 1.0 = white noise)
    pub spectral_flatness: f32,
    
    /// Spectral flux (rate of change)
    pub spectral_flux: f32,
    
    /// Zero crossing rate
    pub zero_crossing_rate: f32,
    
    /// Number of frames used in estimation
    pub frame_count: usize,
    
    /// Confidence level (0.0-1.0)
    pub confidence: f32,
}

impl Default for NoiseStatistics {
    fn default() -> Self {
        Self {
            mean_db: -60.0,
            std_dev_db: 0.0,
            min_db: -80.0,
            max_db: -40.0,
            median_db: -60.0,
            spectral_centroid: 0.0,
            spectral_flatness: 0.5,
            spectral_flux: 0.0,
            zero_crossing_rate: 0.0,
            frame_count: 0,
            confidence: 0.0,
        }
    }
}

// ============================================================================
// Ring Buffer for History (Memory Optimization)
// ============================================================================

/// Ring buffer for efficient spectrum history storage
struct SpectrumRingBuffer {
    buffer: Vec<Vec<f32>>,
    capacity: usize,
    write_index: usize,
    size: usize,
}

impl SpectrumRingBuffer {
    fn new(capacity: usize, spectrum_size: usize) -> Self {
        let buffer = (0..capacity)
            .map(|_| vec![0.0; spectrum_size])
            .collect();
        
        Self {
            buffer,
            capacity,
            write_index: 0,
            size: 0,
        }
    }
    
    fn push(&mut self, spectrum: &[f32]) {
        self.buffer[self.write_index].copy_from_slice(spectrum);
        self.write_index = (self.write_index + 1) % self.capacity;
        self.size = self.size.saturating_add(1).min(self.capacity);
    }
    
    fn iter(&self) -> impl Iterator<Item = &[f32]> {
        let start = if self.size < self.capacity {
            0
        } else {
            self.write_index
        };
        
        (0..self.size).map(move |i| {
            let idx = (start + i) % self.capacity;
            &self.buffer[idx][..]
        })
    }
    
    fn clear(&mut self) {
        self.write_index = 0;
        self.size = 0;
    }
    
    fn len(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Noise Profile
// ============================================================================

/// Noise profile for spectral subtraction
/// 
/// # Thread Safety
/// This struct is not thread-safe. For concurrent access, wrap in Arc<Mutex<_>> or similar.
pub struct NoiseProfile {
    /// Frequency bin magnitudes representing noise spectrum
    spectrum: Vec<f32>,
    
    /// Number of frequency bins
    num_bins: usize,
    
    /// Sample rate in Hz
    sample_rate: f32,
    
    /// FFT size used for spectrum
    fft_size: usize,
    
    /// Smoothing factor for exponential averaging (0.0-1.0)
    smoothing_factor: f32,
    
    /// History buffer using ring buffer for efficiency
    history: SpectrumRingBuffer,
    
    /// Current statistics
    statistics: NoiseStatistics,
    
    /// Is the profile locked (manual mode)?
    is_locked: bool,
    
    /// Frame counter
    frame_count: usize,
    
    /// Minimum spectrum values (per bin) over history
    min_spectrum: Vec<f32>,
    
    /// Maximum spectrum values (per bin) over history  
    max_spectrum: Vec<f32>,
    
    /// Previous spectrum for flux calculation
    prev_spectrum: Vec<f32>,
}

impl NoiseProfile {
    /// Create a new noise profile
    pub fn new(num_bins: usize, sample_rate: f32, fft_size: usize) -> Self {
        let initial_magnitude = db_to_linear(-50.0);
        
        Self {
            spectrum: vec![initial_magnitude; num_bins],
            num_bins,
            sample_rate,
            fft_size,
            smoothing_factor: DEFAULT_SMOOTHING_FACTOR,
            history: SpectrumRingBuffer::new(MAX_FRAMES_FOR_ESTIMATION, num_bins),
            statistics: NoiseStatistics::default(),
            is_locked: false,
            frame_count: 0,
            min_spectrum: vec![f32::MAX; num_bins],
            max_spectrum: vec![0.0; num_bins],
            prev_spectrum: vec![initial_magnitude; num_bins],
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(
        num_bins: usize,
        sample_rate: f32,
        fft_size: usize,
        smoothing_factor: f32,
        max_history_size: usize,
    ) -> Result<Self, NoiseProfileError> {
        // Validate configuration
        Self::validate_config(num_bins, sample_rate, fft_size, smoothing_factor, max_history_size)?;
        
        let mut profile = Self::new(num_bins, sample_rate, fft_size);
        profile.smoothing_factor = smoothing_factor;
        profile.history = SpectrumRingBuffer::new(max_history_size, num_bins);
        
        Ok(profile)
    }
    
    /// Validate configuration parameters
    pub fn validate_config(
        num_bins: usize,
        sample_rate: f32,
        fft_size: usize,
        smoothing_factor: f32,
        max_history_size: usize,
    ) -> Result<(), NoiseProfileError> {
        // Smoothing factor: 0.0 = no smoothing, 1.0 = no update
        if !(0.0..=1.0).contains(&smoothing_factor) {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: format!("Smoothing factor {} out of range [0.0, 1.0]", smoothing_factor),
            });
        }
        
        // Special handling for smoothing factor of 1.0
        if smoothing_factor == 1.0 {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: "Smoothing factor of 1.0 would prevent any updates".to_string(),
            });
        }
        
        if max_history_size < MIN_FRAMES_FOR_ESTIMATION {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: format!("History size {} less than minimum {}", 
                                max_history_size, MIN_FRAMES_FOR_ESTIMATION),
            });
        }
        
        // Validate FFT size and bin relationship
        if num_bins > fft_size / 2 + 1 {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: format!("Number of bins {} exceeds maximum for FFT size {} (max: {})", 
                                num_bins, fft_size, fft_size / 2 + 1),
            });
        }
        
        // Validate sample rate
        if sample_rate <= 0.0 {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: format!("Invalid sample rate: {}", sample_rate),
            });
        }
        
        Ok(())
    }
    
    /// Get the current noise spectrum
    #[inline]
    pub fn spectrum(&self) -> &[f32] {
        &self.spectrum
    }
    
    /// Get current statistics
    #[inline]
    pub fn statistics(&self) -> &NoiseStatistics {
        &self.statistics
    }
    
    /// Check if profile is locked
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }
    
    /// Lock the profile (prevent updates)
    pub fn lock(&mut self) {
        self.is_locked = true;
    }
    
    /// Unlock the profile (allow updates)
    pub fn unlock(&mut self) {
        self.is_locked = false;
    }
    
    /// Get confidence level (0.0-1.0)
    #[inline]
    pub fn confidence(&self) -> f32 {
        self.statistics.confidence
    }
    
    /// Update profile with a new spectrum frame
    pub fn update(&mut self, magnitude_spectrum: &[f32]) -> Result<(), NoiseProfileError> {
        if self.is_locked {
            return Err(NoiseProfileError::ProfileLocked);
        }
        
        if magnitude_spectrum.len() != self.num_bins {
            return Err(NoiseProfileError::InvalidConfiguration {
                message: format!("Spectrum size {} doesn't match profile bins {}", 
                                magnitude_spectrum.len(), self.num_bins),
            });
        }
        
        // Add to history using ring buffer
        self.history.push(magnitude_spectrum);
        
        // Update min/max tracking
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            // Filter out NaN/Inf values
            if mag.is_finite() {
                self.min_spectrum[i] = self.min_spectrum[i].min(mag);
                self.max_spectrum[i] = self.max_spectrum[i].max(mag);
            }
        }
        
        // Update spectrum using exponential averaging
        if self.frame_count == 0 {
            self.spectrum.copy_from_slice(magnitude_spectrum);
            self.prev_spectrum.copy_from_slice(magnitude_spectrum);
        } else {
            // Store previous for flux calculation
            self.prev_spectrum.copy_from_slice(&self.spectrum);
            
            let alpha = self.smoothing_factor;
            let one_minus_alpha = 1.0 - alpha;
            
            for (i, &new_val) in magnitude_spectrum.iter().enumerate() {
                if new_val.is_finite() {
                    self.spectrum[i] = alpha * self.spectrum[i] + one_minus_alpha * new_val;
                }
            }
        }
        
        self.frame_count += 1;
        
        // Update statistics
        self.update_statistics();
        
        Ok(())
    }
    
    /// Calculate spectral gate with pre-allocated gain buffer (performance optimization)
    pub fn calculate_spectral_gate_into(
        &self, 
        magnitude_spectrum: &[f32], 
        gate_threshold: f32, 
        gains: &mut [f32]
    ) -> Result<(), NoiseProfileError> {
        if gains.len() != magnitude_spectrum.len() {
            return Err(NoiseProfileError::BufferSizeMismatch {
                expected: magnitude_spectrum.len(),
                actual: gains.len(),
            });
        }
        
        gains.fill(1.0);
        
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            if i < self.spectrum.len() && mag.is_finite() {
                let noise_level = self.spectrum[i];
                let threshold = noise_level * gate_threshold;
                
                if mag < threshold {
                    gains[i] = 0.0;
                } else {
                    let ratio = mag / threshold;
                    
                    if ratio < 1.5 {
                        gains[i] = (ratio - 1.0) * 2.0;
                    } else {
                        gains[i] = 1.0;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate spectral gate (allocating version for compatibility)
    pub fn calculate_spectral_gate(&self, magnitude_spectrum: &[f32], gate_threshold: f32) -> Vec<f32> {
        let mut gains = vec![1.0; magnitude_spectrum.len()];
        let _ = self.calculate_spectral_gate_into(magnitude_spectrum, gate_threshold, &mut gains);
        gains
    }
    
    /// Enhanced VAD with multiple features
    pub fn is_speech(&self, magnitude_spectrum: &[f32], threshold_db: Option<f32>) -> bool {
        let threshold = threshold_db.unwrap_or(DEFAULT_VAD_THRESHOLD_DB);
        
        // Calculate energy ratio
        let input_energy: f32 = magnitude_spectrum.iter()
            .filter(|x| x.is_finite())
            .map(|&x| x * x)
            .sum();
        
        // Frequency-weighted energy (emphasize speech frequencies 300-3400 Hz)
        let weighted_energy = self.calculate_weighted_energy(magnitude_spectrum);
        
        let noise_energy: f32 = self.spectrum.iter()
            .filter(|x| x.is_finite())
            .map(|&x| x * x)
            .sum();
        
        if noise_energy <= DENORMAL_PREVENTION {
            return true; // Conservative: assume speech if no noise estimate
        }
        
        let energy_ratio_db = 10.0 * (input_energy / noise_energy).log10();
        let weighted_ratio_db = 10.0 * (weighted_energy / noise_energy).log10();
        
        // Calculate spectral flux
        let spectral_flux = self.calculate_spectral_flux(magnitude_spectrum);
        
        // Calculate zero crossing rate
        let zcr = self.estimate_zero_crossing_rate(magnitude_spectrum);
        
        // Combine multiple features for robust detection
        let energy_check = energy_ratio_db > threshold;
        let weighted_check = weighted_ratio_db > threshold * 0.8;
        let flux_check = spectral_flux > 0.3;
        let zcr_check = zcr > 0.1 && zcr < 0.5; // Speech typically has moderate ZCR
        
        // Decision logic: at least 2 checks must pass
        let checks_passed = [energy_check, weighted_check, flux_check, zcr_check]
            .iter()
            .filter(|&&x| x)
            .count();
        
        checks_passed >= 2
    }
    
    /// Calculate frequency-weighted energy (emphasize speech frequencies)
    fn calculate_weighted_energy(&self, magnitude_spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            if mag.is_finite() {
                let freq = (i as f32 * self.sample_rate) / (self.fft_size as f32);
                
                // Weight function: emphasize 300-3400 Hz (speech range)
                let weight = if freq >= 300.0 && freq <= 3400.0 {
                    1.5
                } else if freq < 300.0 {
                    0.5
                } else {
                    0.3
                };
                
                weighted_sum += mag * mag * weight;
            }
        }
        
        weighted_sum
    }
    
    /// Calculate spectral flux
    fn calculate_spectral_flux(&self, magnitude_spectrum: &[f32]) -> f32 {
        let mut flux = 0.0;
        let mut count = 0;
        
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            if i < self.prev_spectrum.len() && mag.is_finite() && self.prev_spectrum[i].is_finite() {
                let diff = (mag - self.prev_spectrum[i]).max(0.0); // Only positive differences
                flux += diff * diff;
                count += 1;
            }
        }
        
        if count > 0 {
            (flux / count as f32).sqrt()
        } else {
            0.0
        }
    }
    
    /// Estimate zero crossing rate from spectrum
    fn estimate_zero_crossing_rate(&self, magnitude_spectrum: &[f32]) -> f32 {
        // Approximate ZCR from spectral characteristics
        // Higher frequency content correlates with higher ZCR
        let mut weighted_freq = 0.0;
        let mut total_energy = 0.0;
        
        for (i, &mag) in magnitude_spectrum.iter().enumerate() {
            if mag.is_finite() {
                let freq = (i as f32) / (self.num_bins as f32);
                weighted_freq += freq * mag;
                total_energy += mag;
            }
        }
        
        if total_energy > DENORMAL_PREVENTION {
            (weighted_freq / total_energy).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Update statistics based on current spectrum (optimized)
    fn update_statistics(&mut self) {
        // Calculate statistics in dB space for correct standard deviation
        let mut db_values: Vec<f32> = Vec::with_capacity(self.spectrum.len());
        let mut linear_sum = 0.0;
        let mut min_linear = f32::MAX;
        let mut max_linear = 0.0_f32;
        
        for &val in self.spectrum.iter() {
            if val.is_finite() && val > DENORMAL_PREVENTION {
                db_values.push(linear_to_db(val));
                linear_sum += val;
                min_linear = min_linear.min(val);
                max_linear = max_linear.max(val);
            }
        }
        
        if db_values.is_empty() {
            self.statistics = NoiseStatistics::default();
            return;
        }
        
        // Mean in dB space
        let mean_db = db_values.iter().sum::<f32>() / db_values.len() as f32;
        self.statistics.mean_db = mean_db;
        
        // Standard deviation in dB space
        let variance_db = db_values.iter()
            .map(|&x| (x - mean_db).powi(2))
            .sum::<f32>() / db_values.len() as f32;
        self.statistics.std_dev_db = variance_db.sqrt();
        
        // Min/Max
        self.statistics.min_db = linear_to_db(min_linear);
        self.statistics.max_db = linear_to_db(max_linear);
        
        // Approximate median using quickselect algorithm (O(n) average)
        self.statistics.median_db = if !db_values.is_empty() {
            Self::approximate_median(&mut db_values)
        } else {
            mean_db
        };
        
        // Calculate spectral features
        self.calculate_spectral_features(linear_sum / self.spectrum.len() as f32);
        
        // Update confidence
        self.update_confidence();
    }
    
    /// Approximate median using quickselect algorithm
    fn approximate_median(values: &mut [f32]) -> f32 {
        let len = values.len();
        if len == 0 {
            return 0.0;
        }
        
        let mid = len / 2;
        
        // Simple selection for small arrays
        if len < 10 {
            values.sort_unstable_by(|a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            return values[mid];
        }
        
        // For larger arrays, use partial sorting (more efficient)
        let (_, median, _) = values.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        *median
    }
    
    /// Calculate spectral features
    fn calculate_spectral_features(&mut self, mean_linear: f32) {
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        let mut log_sum = 0.0;
        let mut flux_sum = 0.0;
        let mut valid_count = 0;
        
        for (i, &mag) in self.spectrum.iter().enumerate() {
            if mag.is_finite() && mag > DENORMAL_PREVENTION {
                let freq = (i as f32 * self.sample_rate) / (self.fft_size as f32);
                weighted_sum += freq * mag;
                magnitude_sum += mag;
                log_sum += mag.ln();
                valid_count += 1;
                
                if i < self.prev_spectrum.len() {
                    let diff = (mag - self.prev_spectrum[i]).abs();
                    flux_sum += diff;
                }
            }
        }
        
        // Spectral centroid
        self.statistics.spectral_centroid = if magnitude_sum > DENORMAL_PREVENTION {
            weighted_sum / magnitude_sum
        } else {
            0.0
        };
        
        // Spectral flatness
        if valid_count > 0 && mean_linear > DENORMAL_PREVENTION {
            let geometric_mean = (log_sum / valid_count as f32).exp();
            self.statistics.spectral_flatness = (geometric_mean / mean_linear).min(1.0);
        } else {
            self.statistics.spectral_flatness = 0.0;
        }
        
        // Spectral flux
        self.statistics.spectral_flux = if valid_count > 0 {
            flux_sum / valid_count as f32
        } else {
            0.0
        };
        
        // Estimate zero crossing rate
        self.statistics.zero_crossing_rate = self.estimate_zero_crossing_rate(&self.spectrum);
        
        self.statistics.frame_count = self.frame_count;
    }
    
    /// Update confidence metric
    fn update_confidence(&mut self) {
        // Confidence based on multiple factors
        let frame_confidence = (self.frame_count as f32 / MIN_FRAMES_FOR_ESTIMATION as f32).min(1.0);
        
        // Lower variance = higher confidence
        let variance_factor = if self.statistics.std_dev_db > 0.0 {
            (-self.statistics.std_dev_db / 10.0).exp()
        } else {
            1.0
        };
        
        // Spectral stability factor
        let stability_factor = 1.0 - self.statistics.spectral_flux.min(1.0);
        
        self.statistics.confidence = (frame_confidence * variance_factor * stability_factor)
            .clamp(0.0, 1.0);
    }
    
    /// Reset the noise profile
    pub fn reset(&mut self) {
        let initial_magnitude = db_to_linear(MIN_NOISE_FLOOR_DB);
        self.spectrum.fill(initial_magnitude);
        self.history.clear();
        self.min_spectrum.fill(f32::MAX);
        self.max_spectrum.fill(0.0);
        self.prev_spectrum.fill(initial_magnitude);
        self.frame_count = 0;
        self.statistics = NoiseStatistics::default();
        self.is_locked = false;
    }
    
    /// Estimate noise from a collection of frames (batch mode)
    pub fn estimate_from_frames(&mut self, frames: &[Vec<f32>]) -> Result<(), NoiseProfileError> {
        if self.is_locked {
            return Err(NoiseProfileError::ProfileLocked);
        }
        
        if frames.len() < MIN_FRAMES_FOR_ESTIMATION {
            return Err(NoiseProfileError::InsufficientData {
                frames_needed: MIN_FRAMES_FOR_ESTIMATION,
                frames_available: frames.len(),
            });
        }
        
        // Clear and reset
        self.reset();
        self.is_locked = false; // Ensure we can update
        
        // Use minimum statistics approach
        for frame in frames.iter() {
            if frame.len() != self.num_bins {
                continue;
            }
            
            self.history.push(frame);
            
            for (i, &mag) in frame.iter().enumerate() {
                if mag.is_finite() {
                    self.min_spectrum[i] = self.min_spectrum[i].min(mag);
                    self.max_spectrum[i] = self.max_spectrum[i].max(mag);
                }
            }
        }
        
        // Use minimum spectrum with safety margin
        const SAFETY_MARGIN: f32 = 1.41; // 3dB
        
        for i in 0..self.num_bins {
            if self.min_spectrum[i].is_finite() && self.min_spectrum[i] != f32::MAX {
                self.spectrum[i] = self.min_spectrum[i] * SAFETY_MARGIN;
            }
        }
        
        self.frame_count = frames.len();
        self.update_statistics();
        
        Ok(())
    }
    
    /// Get noise level at specific frequency
    pub fn noise_at_frequency(&self, frequency: f32) -> f32 {
        let bin = ((frequency * self.fft_size as f32) / self.sample_rate) as usize;
        
        if bin < self.spectrum.len() {
            self.spectrum[bin]
        } else {
            0.0
        }
    }
}

// ============================================================================
// Adaptive Noise Tracker
// ============================================================================

/// Adaptive noise tracker that continuously updates the profile
/// 
/// # Thread Safety
/// Not thread-safe. Wrap in Arc<Mutex<_>> for concurrent use.
pub struct AdaptiveNoiseTracker {
    /// Main noise profile
    profile: NoiseProfile,
    
    /// Speech/noise decision history
    decision_history: VecDeque<bool>,
    
    /// Maximum decision history size
    max_decision_history: usize,
    
    /// Speech probability threshold
    speech_threshold: f32,
    
    /// Noise update rate during speech
    speech_update_rate: f32,
    
    /// Noise update rate during silence
    silence_update_rate: f32,
    
    /// Consecutive silence frames counter
    silence_frames: usize,
    
    /// Minimum silence frames before updating
    min_silence_frames: usize,
}

impl AdaptiveNoiseTracker {
    /// Create a new adaptive noise tracker
    pub fn new(num_bins: usize, sample_rate: f32, fft_size: usize) -> Self {
        Self {
            profile: NoiseProfile::new(num_bins, sample_rate, fft_size),
            decision_history: VecDeque::with_capacity(50),
            max_decision_history: 50,
            speech_threshold: 0.7,
            speech_update_rate: 0.995,
            silence_update_rate: 0.95,
            silence_frames: 0,
            min_silence_frames: 10,
        }
    }
    
    /// Process a frame with pre-allocated gain buffer
    pub fn process_with_gains(
        &mut self, 
        magnitude_spectrum: &[f32],
        gains: &mut [f32]
    ) -> Result<bool, NoiseProfileError> {
        let is_speech = self.profile.is_speech(magnitude_spectrum, None);
        
        // Update decision history
        if self.decision_history.len() >= self.max_decision_history {
            self.decision_history.pop_front();
        }
        self.decision_history.push_back(is_speech);
        
        // Calculate speech probability
        let speech_count = self.decision_history.iter().filter(|&&x| x).count();
        let speech_probability = speech_count as f32 / self.decision_history.len().max(1) as f32;
        
        // Update silence counter
        if is_speech {
            self.silence_frames = 0;
        } else {
            self.silence_frames += 1;
        }
        
        // Adaptive update rate
        let update_rate = if speech_probability > self.speech_threshold {
            self.speech_update_rate
        } else if self.silence_frames >= self.min_silence_frames {
            self.silence_update_rate
        } else {
            0.98
        };
        
        // Update profile
        if !self.profile.is_locked() {
            self.profile.smoothing_factor = update_rate;
            self.profile.update(magnitude_spectrum)?;
        }
        
        // Calculate gains if buffer provided
        if !gains.is_empty() {
            self.profile.calculate_spectral_gate_into(magnitude_spectrum, 1.5, gains)?;
        }
        
        Ok(is_speech)
    }
    
    /// Process a frame (compatibility method)
    pub fn process(&mut self, magnitude_spectrum: &[f32]) -> bool {
        let mut empty_gains = Vec::new();
        self.process_with_gains(magnitude_spectrum, &mut empty_gains)
            .unwrap_or(false)
    }
    
    /// Get the current noise profile
    #[inline]
    pub fn profile(&self) -> &NoiseProfile {
        &self.profile
    }
    
    /// Get mutable access to the profile
    #[inline]
    pub fn profile_mut(&mut self) -> &mut NoiseProfile {
        &mut self.profile
    }
    
    /// Reset the tracker
    pub fn reset(&mut self) {
        self.profile.reset();
        self.decision_history.clear();
        self.silence_frames = 0;
    }
}

// ============================================================================
// Comfort Noise Generator
// ============================================================================

/// Generate comfort noise to fill gaps after aggressive noise reduction
/// 
/// # Thread Safety
/// Not thread-safe due to internal RNG state.
pub struct ComfortNoiseGenerator {
    /// Target noise profile to emulate
    target_profile: Vec<f32>,
    
    /// Random phase for each bin
    phases: Vec<f32>,
    
    /// Gain factor for comfort noise
    gain: f32,
    
    /// High-quality PRNG for better noise characteristics
    rng: SmallRng,
}

impl ComfortNoiseGenerator {
    /// Create a new comfort noise generator
    pub fn new(num_bins: usize) -> Self {
        Self {
            target_profile: vec![0.0; num_bins],
            phases: vec![0.0; num_bins],
            gain: 0.1,
            rng: SmallRng::seed_from_u64(12345),
        }
    }
    
    /// Create with custom seed
    pub fn with_seed(num_bins: usize, seed: u64) -> Self {
        Self {
            target_profile: vec![0.0; num_bins],
            phases: vec![0.0; num_bins],
            gain: 0.1,
            rng: SmallRng::seed_from_u64(seed),
        }
    }
    
    /// Update target profile from noise estimate
    pub fn set_target_profile(&mut self, profile: &[f32]) {
        self.target_profile.resize(profile.len(), 0.0);
        self.target_profile.copy_from_slice(profile);
        self.phases.resize(profile.len(), 0.0);
    }
    
    /// Generate comfort noise spectrum
    pub fn generate(&mut self) -> Vec<(f32, f32)> {
        let mut spectrum = Vec::with_capacity(self.target_profile.len());
        
        for (i, &target_mag) in self.target_profile.iter().enumerate() {
            // High-quality random phase
            let phase = self.rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
            self.phases[i] = phase;
            
            // Add slight magnitude variation for more natural sound
            let magnitude_variation = self.rng.gen_range(0.9..1.1);
            let magnitude = target_mag * self.gain * magnitude_variation;
            
            spectrum.push((magnitude, phase));
        }
        
        spectrum
    }
    
    /// Generate comfort noise with pre-allocated buffers
    pub fn generate_into(&mut self, magnitudes: &mut [f32], phases: &mut [f32]) {
        let len = magnitudes.len().min(phases.len()).min(self.target_profile.len());
        
        for i in 0..len {
            phases[i] = self.rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
            let magnitude_variation = self.rng.gen_range(0.9..1.1);
            magnitudes[i] = self.target_profile[i] * self.gain * magnitude_variation;
        }
    }
    
    /// Set comfort noise gain (0.0-1.0)
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.0, 1.0);
    }
}
