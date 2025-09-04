//! Real-time noise gate implementation for audio processing
//! 
//! This module provides a high-quality noise gate with features including:
//! - Smooth envelope following with separate attack/release times
//! - Hold time to prevent chattering
//! - Lookahead for anticipating transients
//! - Soft knee for transparent gating
//! - Hysteresis to prevent rapid state changes
//! - Variable ratio for expansion vs hard gating

use crate::utils::{
    db_to_linear, linear_to_db, EnvelopeFollower, DualRateParameterSmoother, 
    lerp, DENORMAL_PREVENTION
};
use crate::ring_buffer::{Producer, Consumer, spsc_ring_buffer};

/// Lookahead buffer combining Producer and Consumer for delay line functionality
struct LookaheadBuffer {
    producer: Producer,
    consumer: Consumer,
}

impl LookaheadBuffer {
    fn new(size: usize) -> Option<Self> {
        match spsc_ring_buffer(size) {
            Ok((producer, consumer)) => Some(LookaheadBuffer { producer, consumer }),
            Err(_) => None,
        }
    }
    
    fn write(&mut self, samples: &[f32]) -> Result<(), ()> {
        self.producer.write(samples).map(|_| ()).map_err(|_| ())
    }
    
    fn read(&mut self, output: &mut [f32]) -> Result<(), ()> {
        self.consumer.read(output).map(|_| ()).map_err(|_| ())
    }
    
    fn reset(&mut self) {
        // Reset by draining all available data
        let mut drain_buffer = vec![0.0; 1024];
        while self.consumer.available() > 0 {
            let _ = self.consumer.read(&mut drain_buffer);
        }
    }
}

/// Default values for noise gate parameters
pub mod defaults {
    pub const THRESHOLD_DB: f32 = -40.0;
    pub const RATIO: f32 = 4.0;  // Reduced from 10.0 for more natural gating
    pub const ATTACK_MS: f32 = 2.0;  // Faster attack to preserve transients
    pub const RELEASE_MS: f32 = 200.0;  // Increased from 100ms for smoother fade
    pub const HOLD_MS: f32 = 30.0;  // Increased from 10ms to prevent stuttering
    pub const LOOKAHEAD_MS: f32 = 5.0;
    pub const HYSTERESIS_DB: f32 = 3.0;
    pub const KNEE_WIDTH_DB: f32 = 2.0;
    pub const RMS_WINDOW_MS: f32 = 10.0;
}

/// Noise gate detection mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMode {
    /// Peak detection (instantaneous)
    Peak,
    /// RMS detection (average power)
    Rms,
    /// Mixed mode (combination of peak and RMS)
    Mixed { peak_weight: f32 },
}

/// Noise gate state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateState {
    /// Gate is closed (signal below threshold)
    Closed,
    /// Gate is opening (attack phase)
    Opening,
    /// Gate is open (signal above threshold)
    Open,
    /// Gate is holding open
    Holding,
    /// Gate is closing (release phase)
    Closing,
}

/// Configuration for the noise gate
#[derive(Debug, Clone)]
pub struct NoiseGateConfig {
    /// Threshold in decibels
    pub threshold_db: f32,
    
    /// Expansion ratio (1.0 = no gating, ∞ = hard gate)
    pub ratio: f32,
    
    /// Attack time in milliseconds
    pub attack_ms: f32,
    
    /// Release time in milliseconds
    pub release_ms: f32,
    
    /// Hold time in milliseconds
    pub hold_ms: f32,
    
    /// Lookahead time in milliseconds
    pub lookahead_ms: f32,
    
    /// Hysteresis in dB (difference between open and close thresholds)
    pub hysteresis_db: f32,
    
    /// Enable soft knee
    pub soft_knee: bool,
    
    /// Soft knee width in dB
    pub knee_width_db: f32,
    
    /// Detection mode
    pub detection_mode: DetectionMode,
    
    /// RMS window size in milliseconds (for RMS mode)
    pub rms_window_ms: f32,
    
    /// Gate attack time in milliseconds (for gain smoothing)
    pub gate_attack_ms: f32,
    
    /// Gate release time in milliseconds (for gain smoothing)
    pub gate_release_ms: f32,
    
    /// Sample rate
    pub sample_rate: f32,
}

impl Default for NoiseGateConfig {
    fn default() -> Self {
        Self {
            threshold_db: defaults::THRESHOLD_DB,
            ratio: defaults::RATIO,
            attack_ms: defaults::ATTACK_MS,
            release_ms: defaults::RELEASE_MS,
            hold_ms: defaults::HOLD_MS,
            lookahead_ms: defaults::LOOKAHEAD_MS,
            hysteresis_db: defaults::HYSTERESIS_DB,
            soft_knee: false,
            knee_width_db: defaults::KNEE_WIDTH_DB,
            detection_mode: DetectionMode::Peak,
            rms_window_ms: defaults::RMS_WINDOW_MS,
            gate_attack_ms: 5.0,  // 5ms for smoother opening without cutting transients
            gate_release_ms: 150.0,  // 150ms for natural fade, prevents choppy sound
            sample_rate: 48000.0,
        }
    }
}

impl NoiseGateConfig {
    /// Create config with specific sample rate
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }
    
    /// Builder method for threshold
    pub fn with_threshold_db(mut self, threshold_db: f32) -> Self {
        self.threshold_db = threshold_db;
        self
    }
    
    /// Builder method for times
    pub fn with_times(mut self, attack_ms: f32, release_ms: f32, hold_ms: f32) -> Self {
        self.attack_ms = attack_ms;
        self.release_ms = release_ms;
        self.hold_ms = hold_ms;
        self
    }
    
    /// Builder method for soft knee
    pub fn with_soft_knee(mut self, enabled: bool, width_db: f32) -> Self {
        self.soft_knee = enabled;
        self.knee_width_db = width_db;
        self
    }
}

/// Real-time noise gate processor
pub struct NoiseGate {
    // Configuration
    config: NoiseGateConfig,
    
    // State
    state: GateState,
    
    // Threshold values
    open_threshold_linear: f32,
    close_threshold_linear: f32,
    
    // Envelope follower for signal detection
    envelope_follower: EnvelopeFollower,
    
    // Gain smoothing with dual rates
    gain_smoother: DualRateParameterSmoother,
    current_gain: f32,
    target_gain: f32,
    
    // Hold timer
    hold_samples: usize,
    hold_counter: usize,
    
    // Lookahead buffer (if enabled)
    lookahead_buffer: Option<LookaheadBuffer>,
    lookahead_samples: usize,
    
    // RMS detection buffer (if using RMS mode)
    rms_buffer: Vec<f32>,
    rms_index: usize,
    rms_sum: f32,
    
    // Statistics
    pub total_samples_processed: u64,
    pub gate_open_count: u64,
    pub gate_close_count: u64,
    pub time_open_samples: u64,
    pub time_closed_samples: u64,
}

impl NoiseGate {
    /// Create a new noise gate with given configuration
    pub fn new(config: NoiseGateConfig) -> Self {
        let open_threshold_linear = db_to_linear(config.threshold_db);
        let close_threshold_linear = db_to_linear(config.threshold_db - config.hysteresis_db);
        
        let envelope_follower = EnvelopeFollower::new(
            config.attack_ms,
            config.release_ms,
            config.sample_rate,
        );
        
        // Start with gate closed (gain = 0)
        let gain_smoother = DualRateParameterSmoother::new(
            0.0,
            config.gate_attack_ms,
            config.gate_release_ms,
            config.sample_rate,
        );
        
        let hold_samples = ((config.hold_ms * 0.001 * config.sample_rate) as usize).max(1);
        
        // Setup lookahead if needed
        let (lookahead_buffer, lookahead_samples) = if config.lookahead_ms > 0.0 {
            let samples = ((config.lookahead_ms * 0.001 * config.sample_rate) as usize).max(1);
            let buffer_size = samples.next_power_of_two() * 2; // Extra space for safety
            (
                LookaheadBuffer::new(buffer_size),
                samples,
            )
        } else {
            (None, 0)
        };
        
        // Setup RMS buffer if needed
        let rms_window_samples = ((config.rms_window_ms * 0.001 * config.sample_rate) as usize).max(1);
        let rms_buffer = vec![0.0; rms_window_samples];
        
        Self {
            config,
            state: GateState::Closed,
            open_threshold_linear,
            close_threshold_linear,
            envelope_follower,
            gain_smoother,
            current_gain: 0.0,
            target_gain: 0.0,
            hold_samples,
            hold_counter: 0,
            lookahead_buffer,
            lookahead_samples,
            rms_buffer,
            rms_index: 0,
            rms_sum: 0.0,
            total_samples_processed: 0,
            gate_open_count: 0,
            gate_close_count: 0,
            time_open_samples: 0,
            time_closed_samples: 0,
        }
    }
    
    /// Process a single sample
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // 1) Detector on the current (non-delayed) sample
        let detection_signal = self.get_detection_signal(input);
        let envelope = self.envelope_follower.process(detection_signal);
        
        // 2) Compute & set target gain
        self.target_gain = self.calculate_gain(envelope);
        self.gain_smoother.set_target(self.target_gain);
        
        // 3) Advance gain smoother, then update state
        self.current_gain = self.gain_smoother.next();
        self.update_state(envelope);

        // 4) Apply gain to delayed audio if lookahead enabled
        let output = if let Some(ref mut buffer) = self.lookahead_buffer {
            // Write current sample to delay line
            buffer.write(&[input]).ok();

            // Read delayed sample to apply gain to
            if self.total_samples_processed >= self.lookahead_samples as u64 {
                // Use pre-allocated temp buffer instead of vec
                let mut delayed = [0.0; 1];
                buffer.read(&mut delayed).ok();
                delayed[0] * self.current_gain  // Apply gain to DELAYED audio
            } else {
                0.0  // Initial fill period
            }
        } else {
            input * self.current_gain  // No lookahead - apply gain directly
        };

        self.update_statistics();

        output + DENORMAL_PREVENTION
    }

    /// Process a buffer of samples
    pub fn process_buffer(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "Input and output buffers must have same length");
        
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process_sample(*inp);
        }
    }
    
    /// Process buffer in-place
    pub fn process_buffer_in_place(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Calculate gain reduction based on ratio
    #[inline]
    fn calculate_ratio_gain(&self, envelope_db: f32) -> f32 {
        if self.config.ratio <= 1.0 {
            1.0
        } else if self.config.ratio >= 100.0 {
            0.0
        } else {
            let distance_db = envelope_db - self.config.threshold_db;
            let output_db = self.config.threshold_db + distance_db * self.config.ratio;
            let gain_db = output_db - envelope_db;
            db_to_linear(gain_db).clamp(0.0, 1.0)
        }
    }
    
    /// Calculate ratio gain in linear domain (fast path for hard knee)
    /// This implements a gate as a high-ratio downward expander
    #[inline]
    fn calculate_ratio_gain_linear(&self, envelope: f32) -> f32 {
        // Early returns for special cases
        if self.config.ratio <= 1.0 { 
            return 1.0;  // No gating
        }
        if envelope >= self.open_threshold_linear { 
            return 1.0;  // Above threshold - fully open
        }
        if self.config.ratio >= 100.0 { 
            return 0.0;  // Infinite ratio - hard gate
        }
        
        // Calculate gain using linear domain math
        // For envelope E <= threshold T with ratio R:
        // gain = (E/T)^(R-1)
        let x = (envelope / self.open_threshold_linear).max(0.0);
        let r_minus_1 = self.config.ratio - 1.0;
        x.powf(r_minus_1).clamp(0.0, 1.0)
    }
    
    /// Get detection signal based on detection mode
    #[inline]
    fn get_detection_signal(&mut self, input: f32) -> f32 {
        match self.config.detection_mode {
            DetectionMode::Peak => input.abs(),
            DetectionMode::Rms => {
                // Update RMS buffer
                let old_value = self.rms_buffer[self.rms_index];
                let new_value = input * input;
                
                self.rms_buffer[self.rms_index] = new_value;
                self.rms_sum += new_value - old_value;
                self.rms_index = (self.rms_index + 1) % self.rms_buffer.len();
                
                // Calculate RMS
                (self.rms_sum.max(0.0) / self.rms_buffer.len() as f32).sqrt()
            }
            DetectionMode::Mixed { peak_weight } => {
                let peak = input.abs();
                
                // Update RMS calculation
                let old_value = self.rms_buffer[self.rms_index];
                let new_value = input * input;
                self.rms_buffer[self.rms_index] = new_value;
                self.rms_sum += new_value - old_value;
                self.rms_index = (self.rms_index + 1) % self.rms_buffer.len();
                let rms = (self.rms_sum.max(0.0) / self.rms_buffer.len() as f32).sqrt();
                
                // Weighted mix
                peak * peak_weight + rms * (1.0 - peak_weight)
            }
        }
    }
    
    /// Update gate state based on envelope and hysteresis
    #[inline]
    fn update_state(&mut self, envelope: f32) {
        self.state = match self.state {
            GateState::Closed => {
                if envelope > self.open_threshold_linear {
                    self.gate_open_count += 1;
                    GateState::Opening
                } else {
                    GateState::Closed
                }
            }
            GateState::Opening => {
                if self.gain_smoother.is_settled(0.01) || self.current_gain > 0.9 {
                    self.hold_counter = 0;
                    GateState::Open
                } else {
                    GateState::Opening
                }
            }
            GateState::Open => {
                if envelope < self.close_threshold_linear {
                    if self.hold_samples > 0 {
                        GateState::Holding
                    } else {
                        self.gate_close_count += 1;
                        GateState::Closing
                    }
                } else {
                    GateState::Open
                }
            }
            GateState::Holding => {
                self.hold_counter += 1;
                if envelope > self.close_threshold_linear {
                    // Signal came back up during hold
                    self.hold_counter = 0;
                    GateState::Open
                } else if self.hold_counter >= self.hold_samples {
                    // Hold time expired
                    self.gate_close_count += 1;
                    GateState::Closing
                } else {
                    GateState::Holding
                }
            }
            GateState::Closing => {
                if envelope > self.open_threshold_linear {
                    // Signal came back up during close
                    self.gate_open_count += 1;
                    GateState::Opening
                } else if self.gain_smoother.is_settled(0.01) || self.current_gain < 0.1 {
                    GateState::Closed
                } else {
                    GateState::Closing
                }
            }
        };
    }
    
    /// Calculate gain based on envelope and threshold
    #[inline]
    fn calculate_gain(&self, envelope: f32) -> f32 {
        if self.config.soft_knee {
            // Soft knee requires dB calculations
            let envelope_db = linear_to_db(envelope);
            self.calculate_soft_knee_gain(envelope_db)
        } else {
            // Hard knee can use fast linear domain calculation
            self.calculate_ratio_gain_linear(envelope)
        }
    }
    
    /// Calculate gain with soft knee
    fn calculate_soft_knee_gain(&self, envelope_db: f32) -> f32 {
        let knee_start = self.config.threshold_db - self.config.knee_width_db / 2.0;
        let knee_end = self.config.threshold_db + self.config.knee_width_db / 2.0;

        if envelope_db >= knee_end {
            1.0
        } else if envelope_db <= knee_start {
            self.calculate_ratio_gain(envelope_db)  // Use helper method
        } else {
            // Within knee region - smooth transition
            let knee_position = (envelope_db - knee_start) / self.config.knee_width_db;
            let t = 0.5 * (1.0 - (knee_position * std::f32::consts::PI).cos());
            let below_gain = self.calculate_ratio_gain(knee_start);
            lerp(below_gain, 1.0, t)
        }
    }

    /// Update statistics
    #[inline]
    fn update_statistics(&mut self) {
        self.total_samples_processed += 1;
        
        match self.state {
            GateState::Open | GateState::Opening | GateState::Holding => {
                self.time_open_samples += 1;
            }
            GateState::Closed | GateState::Closing => {
                self.time_closed_samples += 1;
            }
        }
    }
    
    /// Get current state
    pub fn state(&self) -> GateState {
        self.state
    }
    
    /// Get current gain in linear (0.0-1.0)
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }
    
    /// Get current gain in dB
    pub fn current_gain_db(&self) -> f32 {
        linear_to_db(self.current_gain)
    }
    
    /// Get current envelope level
    pub fn envelope(&self) -> f32 {
        self.envelope_follower.current()
    }
    
    /// Get current envelope level in dB
    pub fn envelope_db(&self) -> f32 {
        linear_to_db(self.envelope())
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: NoiseGateConfig) {
        self.open_threshold_linear = db_to_linear(config.threshold_db);
        self.close_threshold_linear = db_to_linear(config.threshold_db - config.hysteresis_db);
        
        self.envelope_follower.set_times(
            config.attack_ms,
            config.release_ms,
            config.sample_rate,
        );
        
        self.hold_samples = ((config.hold_ms * 0.001 * config.sample_rate) as usize).max(1);
        
        // Update lookahead if changed
        if config.lookahead_ms != self.config.lookahead_ms {
            if config.lookahead_ms > 0.0 {
                let samples = ((config.lookahead_ms * 0.001 * config.sample_rate) as usize).max(1);
                let buffer_size = samples.next_power_of_two() * 2;
                self.lookahead_buffer = LookaheadBuffer::new(buffer_size);
                self.lookahead_samples = samples;
            } else {
                self.lookahead_buffer = None;
                self.lookahead_samples = 0;
            }
        }
        
        // Update RMS buffer if window size changed
        if config.rms_window_ms != self.config.rms_window_ms {
            let rms_window_samples = ((config.rms_window_ms * 0.001 * config.sample_rate) as usize).max(1);
            self.rms_buffer = vec![0.0; rms_window_samples];
            self.rms_index = 0;
            self.rms_sum = 0.0;
        }
        
        self.config = config;
    }
    
    pub fn reset(&mut self) {
        self.state = GateState::Closed;
        self.envelope_follower.reset();
        self.gain_smoother.reset(0.0);
        self.current_gain = 0.0;
        self.target_gain = 0.0;
        self.hold_counter = 0;

        if let Some(ref mut buffer) = self.lookahead_buffer {
            buffer.reset();
        }

        self.rms_buffer.fill(0.0);
        self.rms_index = 0;
        self.rms_sum = 0.0;
    }

    /// Get statistics
    pub fn get_stats(&self) -> NoiseGateStats {
        NoiseGateStats {
            total_samples: self.total_samples_processed,
            gate_open_count: self.gate_open_count,
            gate_close_count: self.gate_close_count,
            time_open_ms: (self.time_open_samples as f32 / self.config.sample_rate * 1000.0),
            time_closed_ms: (self.time_closed_samples as f32 / self.config.sample_rate * 1000.0),
            current_state: self.state,
            current_gain_db: self.current_gain_db(),
            envelope_db: self.envelope_db(),
        }
    }
}

/// Statistics from noise gate
#[derive(Debug, Clone)]
pub struct NoiseGateStats {
    pub total_samples: u64,
    pub gate_open_count: u64,
    pub gate_close_count: u64,
    pub time_open_ms: f32,
    pub time_closed_ms: f32,
    pub current_state: GateState,
    pub current_gain_db: f32,
    pub envelope_db: f32,
}

// ============================================================================
// Stereo Noise Gate
// ============================================================================

/// Stereo noise gate with linked/unlinked processing
pub struct StereoNoiseGate {
    left_gate: NoiseGate,
    right_gate: NoiseGate,
    linked: bool,
    link_mode: LinkMode,
}

/// Stereo link mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkMode {
    /// Use maximum of both channels for detection
    Maximum,
    /// Use average of both channels for detection
    Average,
    /// Use left channel for detection
    LeftOnly,
    /// Use right channel for detection
    RightOnly,
}

impl StereoNoiseGate {
    /// Create a new stereo noise gate
    pub fn new(config: NoiseGateConfig, linked: bool) -> Self {
        let left_gate = NoiseGate::new(config.clone());
        let right_gate = NoiseGate::new(config);
        
        Self {
            left_gate,
            right_gate,
            linked,
            link_mode: LinkMode::Maximum,
        }
    }
    
    /// Process stereo samples
    pub fn process_stereo(&mut self, left_in: f32, right_in: f32) -> (f32, f32) {
        if !self.linked {
            // Unlinked mode - independent processing
            let left_out = self.left_gate.process_sample(left_in);
            let right_out = self.right_gate.process_sample(right_in);
            return (left_out, right_out);
        }
        
        // Linked mode - use combined detection that respects detection mode
        
        // 1) Per-channel detectors (respect detection mode / RMS window)
        let left_detection = self.left_gate.get_detection_signal(left_in);
        let right_detection = self.right_gate.get_detection_signal(right_in);
        
        // 2) Link the detector outputs
        let linked_detection = match self.link_mode {
            LinkMode::Maximum => left_detection.max(right_detection),
            LinkMode::Average => 0.5 * (left_detection + right_detection),
            LinkMode::LeftOnly => left_detection,
            LinkMode::RightOnly => right_detection,
        };
        
        // 3) Process both envelopes with the same linked detector signal
        let envelope = self.left_gate.envelope_follower.process(linked_detection);
        self.right_gate.envelope_follower.process(linked_detection);
        
        // 4) Calculate one gain for both channels
        let gain = self.left_gate.calculate_gain(envelope);
        
        // Update both gates with the same gain
        for gate in [&mut self.left_gate, &mut self.right_gate].iter_mut() {
            gate.target_gain = gain;
            gate.gain_smoother.set_target(gain);
            gate.current_gain = gate.gain_smoother.next();
            gate.update_state(envelope);
        }
        
        // 5) Apply lookahead per channel (identical to mono path)
        let left_out = if let Some(ref mut buffer) = self.left_gate.lookahead_buffer {
            // Write current sample to lookahead buffer
            buffer.write(&[left_in]).ok();
            
            // If we have enough samples buffered, read the delayed sample
            if self.left_gate.total_samples_processed >= self.left_gate.lookahead_samples as u64 {
                let mut delayed = [0.0; 1];
                buffer.read(&mut delayed).ok();
                delayed[0] * self.left_gate.current_gain
            } else {
                0.0
            }
        } else {
            left_in * self.left_gate.current_gain
        };
        
        let right_out = if let Some(ref mut buffer) = self.right_gate.lookahead_buffer {
            // Write current sample to lookahead buffer
            buffer.write(&[right_in]).ok();
            
            // If we have enough samples buffered, read the delayed sample
            if self.right_gate.total_samples_processed >= self.right_gate.lookahead_samples as u64 {
                let mut delayed = [0.0; 1];
                buffer.read(&mut delayed).ok();
                delayed[0] * self.right_gate.current_gain
            } else {
                0.0
            }
        } else {
            right_in * self.right_gate.current_gain
        };
        
        // 6) Update statistics
        self.left_gate.update_statistics();
        self.right_gate.update_statistics();
        
        (left_out + DENORMAL_PREVENTION, right_out + DENORMAL_PREVENTION)
    }
    
    /// Set link mode
    pub fn set_link_mode(&mut self, mode: LinkMode) {
        self.link_mode = mode;
    }
    
    /// Set linked status
    pub fn set_linked(&mut self, linked: bool) {
        self.linked = linked;
    }
}

    #[cfg(test)]
    mod tests {
        use super::*;
        
        #[test]
        fn test_noise_gate_creation() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let gate = NoiseGate::new(config);
            
            assert_eq!(gate.state(), GateState::Closed);
            assert_eq!(gate.current_gain(), 0.0);
            assert_eq!(gate.total_samples_processed, 0);
        }
        
        #[test]
        fn test_gate_opens_above_threshold() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.attack_ms = 0.1; // Fast attack for testing
            config.hold_ms = 0.0; // No hold for simplicity
            
            let mut gate = NoiseGate::new(config);
            
            // Signal below threshold - gate stays closed
            let quiet = db_to_linear(-30.0);
            for _ in 0..100 {
                let out = gate.process_sample(quiet);
                assert!(out.abs() < quiet); // Signal is attenuated
            }
            
            // Signal above threshold - gate opens
            let loud = db_to_linear(-10.0);
            // Process more samples to allow envelope to settle
            for _ in 0..5000 {
                gate.process_sample(loud);
            }
            
            assert!(gate.current_gain() > 0.9, "Gate gain is {}", gate.current_gain());
            assert!(matches!(gate.state(), GateState::Open));
        }
        
        #[test]
        fn test_hysteresis() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.hysteresis_db = 5.0;
            config.attack_ms = 0.1;
            config.release_ms = 0.1;
            
            let mut gate = NoiseGate::new(config);
            
            // Open gate with signal above threshold
            let loud = db_to_linear(-10.0);
            for _ in 0..5000 {
                gate.process_sample(loud);
            }
            assert!(matches!(gate.state(), GateState::Open));
            
            // Signal at -22dB (below threshold but above close threshold of -25dB)
            let mid = db_to_linear(-22.0);
            for _ in 0..1000 {
                gate.process_sample(mid);
            }
            // Gate should still be open due to hysteresis
            assert!(matches!(gate.state(), GateState::Open | GateState::Holding));
            
            // Signal well below close threshold
            let quiet = db_to_linear(-30.0);
            for _ in 0..10000 {
                gate.process_sample(quiet);
            }
            assert!(matches!(gate.state(), GateState::Closed));
        }
        
        #[test]
        fn test_hold_time() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.hold_ms = 100.0; // 100ms hold
            config.attack_ms = 0.1;
            config.release_ms = 0.1;
            
            let mut gate = NoiseGate::new(config.clone());
            
            // Open gate
            let loud = db_to_linear(-10.0);
            for _ in 0..5000 {
                gate.process_sample(loud);
            }
            assert!(matches!(gate.state(), GateState::Open));
            
            // Signal drops below threshold
            let quiet = db_to_linear(-30.0);
            let hold_samples = (100.0 * 0.001 * 48000.0) as usize;
            
            // Process for less than hold time
            for _ in 0..hold_samples/2 {
                gate.process_sample(quiet);
            }
            // Should be holding
            assert!(matches!(gate.state(), GateState::Holding));
            
            // Process past hold time
            for _ in 0..hold_samples {
                gate.process_sample(quiet);
            }
            // Should now be closing or closed
            assert!(matches!(gate.state(), GateState::Closing | GateState::Closed));
        }
        
        #[test]
        fn test_soft_knee() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.soft_knee = true;
            config.knee_width_db = 6.0;
            config.attack_ms = 0.1;
            config.release_ms = 0.1;
            
            let mut gate_soft = NoiseGate::new(config.clone());
            
            config.soft_knee = false;
            let mut gate_hard = NoiseGate::new(config);
            
            // Test signal right at threshold
            let at_threshold = db_to_linear(-20.0);
            
            // Process enough samples for envelope to stabilize
            for _ in 0..5000 {
                gate_soft.process_sample(at_threshold);
                gate_hard.process_sample(at_threshold);
            }
            
            // Soft knee should have intermediate gain at threshold
            let soft_gain = gate_soft.current_gain();
            let hard_gain = gate_hard.current_gain();
            
            // At threshold, soft knee should be around 0.5 (middle of transition)
            assert!(soft_gain > 0.3 && soft_gain < 0.7, 
                    "Soft gain {} should be between 0.3 and 0.7", soft_gain);
            // Hard knee should be fully open at threshold
            assert!(hard_gain > 0.9, 
                    "Hard gain {} should be > 0.9", hard_gain);
        }
        
        #[test]
        fn test_rms_detection() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.detection_mode = DetectionMode::Rms;
            config.rms_window_ms = 10.0;
            
            let mut gate = NoiseGate::new(config);
            
            // Generate a sine wave
            let frequency = 1000.0;
            let amplitude = db_to_linear(-15.0); // Above threshold
            let sample_rate = 48000.0;
            
            for i in 0..4800 {
                let phase = 2.0 * std::f32::consts::PI * frequency * (i as f32) / sample_rate;
                let sample = amplitude * phase.sin();
                gate.process_sample(sample);
            }
            
            // RMS of sine wave should trigger gate opening
            assert!(gate.current_gain() > 0.5);
        }
        
        #[test]
        fn test_statistics() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            // Process some samples with gate closed
            let quiet = db_to_linear(-50.0);
            for _ in 0..1000 {
                gate.process_sample(quiet);
            }
            
            // Open gate
            let loud = db_to_linear(-10.0);
            for _ in 0..2000 {
                gate.process_sample(loud);
            }
            
            let stats = gate.get_stats();
            assert_eq!(stats.total_samples, 3000);
            assert!(stats.gate_open_count > 0);
            assert!(stats.time_open_ms > 0.0);
            assert!(stats.time_closed_ms > 0.0);
        }
        
        #[test]
        fn test_stereo_linked_processing() {
            let config = NoiseGateConfig::with_sample_rate(48000.0)
                .with_threshold_db(-20.0);
            let mut stereo_gate = StereoNoiseGate::new(config, true);
            
            // Left channel loud, right channel quiet
            let left = db_to_linear(-10.0);
            let right = db_to_linear(-30.0);
            
            // Process several samples to let gate stabilize
            for _ in 0..1000 {
                stereo_gate.process_stereo(left, right);
            }
            
            // In linked mode, both channels should have same gain
            let left_gain = stereo_gate.left_gate.current_gain();
            let right_gain = stereo_gate.right_gate.current_gain();
            
            assert!((left_gain - right_gain).abs() < 0.001);
            assert!(left_gain > 0.9); // Both should be open
        }
        
        #[test]
        fn test_buffer_processing() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            let input = vec![0.1, -0.2, 0.15, -0.25, 0.3];
            let mut output = vec![0.0; 5];
            
            gate.process_buffer(&input, &mut output);
            
            // Basic check that processing occurred
            for (i, o) in input.iter().zip(output.iter()) {
                assert!(o.abs() <= i.abs()); // Output should be attenuated or same
            }
        }
        
        #[test]
        fn test_in_place_processing() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            let mut buffer = vec![0.1, -0.2, 0.15, -0.25, 0.3];
            let original = buffer.clone();
            
            gate.process_buffer_in_place(&mut buffer);
            
            // Check that processing occurred
            for (orig, processed) in original.iter().zip(buffer.iter()) {
                assert!(processed.abs() <= orig.abs());
            }
        }
        
        #[test]
        fn test_config_update() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            // Process with original config
            let loud = db_to_linear(-10.0);
            for _ in 0..1000 {
                gate.process_sample(loud);
            }
            let gain_before = gate.current_gain();
            
            // Update threshold to make gate close
            let new_config = NoiseGateConfig::with_sample_rate(48000.0)
                .with_threshold_db(-5.0);
            gate.update_config(new_config);
            
            // Process with new config
            for _ in 0..10000 {
                gate.process_sample(loud);
            }
            let gain_after = gate.current_gain();
            
            // Gate should be more closed with higher threshold
            assert!(gain_after < gain_before);
        }
        
        #[test]
        fn test_reset() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            // Open gate and accumulate stats
            let loud = db_to_linear(-10.0);
            for _ in 0..1000 {
                gate.process_sample(loud);
            }
            
            assert!(gate.current_gain() > 0.5);
            assert!(gate.total_samples_processed > 0);
            
            // Reset
            gate.reset();
            
            assert_eq!(gate.state(), GateState::Closed);
            assert_eq!(gate.current_gain(), 0.0);
            assert_eq!(gate.envelope(), 0.0);
        }
        
        #[test]
        fn test_extreme_values() {
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            // Test with extreme values
            let extreme_values = [
                0.0, 1.0, -1.0, 
                f32::MIN_POSITIVE, 
                f32::EPSILON,
                100.0, -100.0,
            ];
            
            for &value in &extreme_values {
                let result = gate.process_sample(value);
                assert!(result.is_finite());
                assert!(!result.is_nan());
            }
        }
        
        #[test]
        fn test_ratio_settings() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.attack_ms = 0.1;
            config.release_ms = 0.1;
            
            // Test with ratio = 1 (no gating)
            config.ratio = 1.0;
            let mut gate_unity = NoiseGate::new(config.clone());
            
            // Test with ratio = infinity (hard gate)
            config.ratio = 1000.0;
            let mut gate_hard = NoiseGate::new(config.clone());
            
            // Test with moderate ratio
            config.ratio = 4.0;
            let mut gate_moderate = NoiseGate::new(config);
            
            // Process signal below threshold
            let quiet = db_to_linear(-30.0);
            for _ in 0..5000 {
                gate_unity.process_sample(quiet);
                gate_hard.process_sample(quiet);
                gate_moderate.process_sample(quiet);
            }
            
            // Unity ratio should have no reduction
            assert!(gate_unity.current_gain() > 0.95, 
                    "Unity gain {} should be > 0.95", gate_unity.current_gain());
            
            // Hard gate should be fully closed  
            assert!(gate_hard.current_gain() < 0.05,
                    "Hard gate gain {} should be < 0.05", gate_hard.current_gain());
            
            // Moderate should be in between
            assert!(gate_moderate.current_gain() > 0.01);
            assert!(gate_moderate.current_gain() < 0.99);
        }
        
        #[test]
        fn test_lookahead() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.lookahead_ms = 5.0;
            config.attack_ms = 50.0; // Slower attack for smoother transition
            config.release_ms = 50.0;
            
            let mut gate = NoiseGate::new(config);
            
            // Process initial silence to fill lookahead buffer
            let lookahead_samples = (5.0 * 0.001 * 48000.0) as usize;
            for _ in 0..lookahead_samples {
                gate.process_sample(0.0);
            }
            
            // Process some quiet samples
            let quiet = db_to_linear(-40.0);
            for _ in 0..100 {
                gate.process_sample(quiet);
            }
            
            // Now process loud signal
            let loud = db_to_linear(-10.0);
            let mut outputs = Vec::new();
            
            // Collect transition samples
            for _ in 0..2000 {
                outputs.push(gate.process_sample(loud));
            }
            
            // Find the maximum difference between consecutive samples
            let mut max_diff = 0.0f32;
            for window in outputs[100..500].windows(2) {
                let diff = (window[1] - window[0]).abs();
                max_diff = max_diff.max(diff);
            }
            
            // With lookahead and 50ms attack, transitions should be smooth
            // But not perfectly smooth due to exponential envelope
            println!("Max diff with lookahead: {}", max_diff);
            assert!(max_diff < 0.5, "Max diff {} should be < 0.5 for lookahead", max_diff);
            
            // Verify gate eventually opens fully
            let final_gain = gate.current_gain();
            assert!(final_gain > 0.9, "Gate should be fully open at end");
        }
        
        #[test]
        fn test_mixed_detection_mode() {
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.detection_mode = DetectionMode::Mixed { peak_weight: 0.5 };
            config.threshold_db = -20.0;
            config.attack_ms = 0.1;
            
            let mut gate = NoiseGate::new(config);
            
            // Create signal with varying dynamics
            let mut signal = Vec::new();
            for i in 0..4800 {
                let t = i as f32 / 48000.0;
                // Mix of sine wave and impulses
                let sine = 0.1 * (2.0 * std::f32::consts::PI * 100.0 * t).sin();
                let impulse = if i % 480 == 0 { 0.5 } else { 0.0 };
                signal.push(sine + impulse);
            }
            
            // Process signal
            for &sample in &signal {
                gate.process_sample(sample);
            }
            
            // Mixed mode should respond to both sustained and transient content
            let stats = gate.get_stats();
            assert!(stats.gate_open_count > 0, "Gate should have opened at least once");
        }
        
        #[test]
        fn test_linked_vs_unlinked_stereo_parity_rms() {
            // Test that linked stereo produces identical gains for both channels
            // while unlinked allows different gains
            
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.detection_mode = DetectionMode::Rms;
            config.rms_window_ms = 10.0;
            config.attack_ms = 1.0;
            config.release_ms = 10.0;
            
            // Create quiet left channel and loud right channel signals
            let quiet_signal = db_to_linear(-30.0); // Below threshold
            let loud_signal = db_to_linear(-10.0);  // Above threshold
            
            // Test linked mode - both channels should get the same gain
            let mut linked_gate = StereoNoiseGate::new(config.clone(), true);
            
            // Process several samples to let RMS window fill
            let mut left_gains_linked = Vec::new();
            let mut right_gains_linked = Vec::new();
            
            for _ in 0..1000 {
                linked_gate.process_stereo(quiet_signal, loud_signal);
            }
            
            // Collect gains over next samples
            for _ in 0..100 {
                linked_gate.process_stereo(quiet_signal, loud_signal);
                left_gains_linked.push(linked_gate.left_gate.current_gain());
                right_gains_linked.push(linked_gate.right_gate.current_gain());
            }
            
            // In linked mode, gains should be identical
            for i in 0..left_gains_linked.len() {
                assert!((left_gains_linked[i] - right_gains_linked[i]).abs() < 1e-6,
                        "Linked mode: gains should be identical. Left: {}, Right: {}", 
                        left_gains_linked[i], right_gains_linked[i]);
            }
            
            // Test unlinked mode - channels can have different gains
            let mut unlinked_gate = StereoNoiseGate::new(config, false);
            
            // Process to steady state
            for _ in 0..1000 {
                unlinked_gate.process_stereo(quiet_signal, loud_signal);
            }
            
            // Check that gains are different
            let left_gain = unlinked_gate.left_gate.current_gain();
            let right_gain = unlinked_gate.right_gate.current_gain();
            
            assert!(left_gain < 0.5, "Unlinked: quiet channel should be gated. Gain: {}", left_gain);
            assert!(right_gain > 0.9, "Unlinked: loud channel should be open. Gain: {}", right_gain);
            assert!((left_gain - right_gain).abs() > 0.4, 
                    "Unlinked mode: gains should differ significantly. Left: {}, Right: {}", 
                    left_gain, right_gain);
        }
        
        #[test]
        fn test_lookahead_with_long_attack_smoothness() {
            // Test that lookahead with long attack produces smooth gain changes
            
            let mut config = NoiseGateConfig::with_sample_rate(48000.0);
            config.threshold_db = -20.0;
            config.lookahead_ms = 5.0;
            config.gate_attack_ms = 50.0; // Long attack for smooth opening
            config.gate_release_ms = 50.0;
            
            let mut gate = NoiseGate::new(config);
            
            // Create a step signal: quiet -> loud
            let quiet = db_to_linear(-30.0);
            let loud = db_to_linear(-10.0);
            
            // Process quiet signal first
            for _ in 0..100 {
                gate.process_sample(quiet);
            }
            
            // Collect gain changes during transition
            let mut gain_differences = Vec::new();
            let mut prev_gain = gate.current_gain();
            
            // Process transition from quiet to loud
            // Need more samples for 50ms attack time at 48kHz (50ms = 2400 samples)
            for i in 0..3000 {
                let input = if i < 50 { quiet } else { loud };
                gate.process_sample(input);
                
                let current_gain = gate.current_gain();
                let diff = (current_gain - prev_gain).abs();
                if diff > 0.0 {
                    gain_differences.push(diff);
                }
                prev_gain = current_gain;
            }
            
            // Check that gain changes are smooth (no large jumps)
            let max_diff = gain_differences.iter().fold(0.0f32, |a, &b| a.max(b));
            let avg_diff = gain_differences.iter().sum::<f32>() / gain_differences.len().max(1) as f32;
            
            // With 50ms attack and proper lookahead, max change per sample should be small
            assert!(max_diff < 0.05, "Max gain change per sample too large: {}", max_diff);
            assert!(avg_diff < 0.02, "Average gain change too large: {}", avg_diff);
            
            // Verify gate eventually opens substantially (after sufficient time for 50ms attack)
            assert!(gate.current_gain() > 0.9, "Gate should be substantially open. Gain: {}", gate.current_gain());
        }
        
        #[test]
        fn test_linear_fast_path_correctness() {
            // Test that linear fast-path produces same results as dB path
            // within acceptable epsilon
            
            let config = NoiseGateConfig::with_sample_rate(48000.0);
            let mut gate = NoiseGate::new(config);
            
            // Test grid of envelope, threshold, and ratio values
            let envelopes = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
            let thresholds_db = [-40.0, -30.0, -20.0, -10.0, -6.0];
            let ratios = [2.0, 4.0, 10.0, 20.0, 50.0];
            
            for &threshold_db in &thresholds_db {
                gate.config.threshold_db = threshold_db;
                gate.open_threshold_linear = db_to_linear(threshold_db);
                
                for &ratio in &ratios {
                    gate.config.ratio = ratio;
                    
                    for &envelope in &envelopes {
                        // Calculate using linear fast-path (hard knee)
                        gate.config.soft_knee = false;
                        let gain_linear = gate.calculate_gain(envelope);
                        
                        // Calculate using dB path (by using calculate_ratio_gain directly)
                        let envelope_db = linear_to_db(envelope);
                        let gain_db = if envelope_db >= threshold_db {
                            1.0
                        } else {
                            gate.calculate_ratio_gain(envelope_db)
                        };
                        
                        // They should match within small epsilon
                        let diff = (gain_linear - gain_db).abs();
                        assert!(diff < 1e-5, 
                                "Linear vs dB mismatch: envelope={}, threshold_db={}, ratio={}, linear={}, db={}, diff={}", 
                                envelope, threshold_db, ratio, gain_linear, gain_db, diff);
                    }
                }
            }
        }
}
