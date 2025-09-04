use std::f32::consts::PI;

// Configuration constants
/// Minimum dB value before clamping to zero linear amplitude
/// Values below this are considered silence to avoid numerical issues
pub const DB_NOISE_FLOOR: f32 = -100.0;

/// Small value added to prevent denormal numbers
pub const DENORMAL_PREVENTION: f32 = 1e-24;

/// Maximum coefficient value for time constants
pub const MAX_TIME_CONSTANT_COEFF: f32 = 0.99999;

// ============================================================================
// Core Conversion Functions
// ============================================================================

/// Convert linear amplitude to decibels
/// Returns -inf for zero or negative values
#[inline]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        -f32::INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Convert decibels to linear amplitude
/// Clamps to 0.0 for very negative dB values
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    if db <= DB_NOISE_FLOOR {
        0.0
    } else {
        10.0_f32.powf(db / 20.0)
    }
}

/// Convert frequency to MIDI note number
#[inline]
pub fn freq_to_midi(freq: f32) -> f32 {
    69.0 + 12.0 * (freq / 440.0).log2()
}

/// Convert MIDI note number to frequency
#[inline]
pub fn midi_to_freq(midi: f32) -> f32 {
    440.0 * 2.0_f32.powf((midi - 69.0) / 12.0)
}

// ============================================================================
// Envelope Following
// ============================================================================

/// Smoothing filter with separate attack and release times
/// Used for envelope following in noise gate
/// 
/// # Thread Safety
/// This type is not thread-safe. Each thread should have its own instance.
pub struct EnvelopeFollower {
    attack_coeff: f32,
    release_coeff: f32,
    current_value: f32,
}

impl EnvelopeFollower {
    /// Create a new envelope follower
    /// attack_ms: Attack time in milliseconds
    /// release_ms: Release time in milliseconds
    /// sample_rate: Sample rate in Hz
    pub fn new(attack_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        Self {
            attack_coeff: calculate_time_constant(attack_ms, sample_rate),
            release_coeff: calculate_time_constant(release_ms, sample_rate),
            current_value: 1e-10,
        }
    }

    /// Process a single sample
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let input = input.abs() + DENORMAL_PREVENTION; // Prevent denormals
        
        let coeff = if input > self.current_value {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        
        self.current_value = input + coeff * (self.current_value - input);
        self.current_value
    }

    /// Reset the envelope to zero
    pub fn reset(&mut self) {
        self.current_value = 0.0;
    }

    /// Update time constants
    pub fn set_times(&mut self, attack_ms: f32, release_ms: f32, sample_rate: f32) {
        self.attack_coeff = calculate_time_constant(attack_ms, sample_rate);
        self.release_coeff = calculate_time_constant(release_ms, sample_rate);
    }
    
    /// Get current envelope value
    #[inline]
    pub fn current(&self) -> f32 {
        self.current_value
    }
}

/// Calculate time constant for exponential smoothing
/// Returns coefficient for: y[n] = x[n] + coeff * (y[n-1] - x[n])
/// Returns 0.0 for instant change (time_ms <= 0)
#[inline]
fn calculate_time_constant(time_ms: f32, sample_rate: f32) -> f32 {
    if time_ms <= 0.0 {
        0.0  // Instant change
    } else {
        // Audio standard: time is 10% to 90% rise time
        // This takes approximately 2.197 time constants for exponential
        let tau = time_ms * 0.001 / 2.197; 
        let coeff = (-1.0 / (tau * sample_rate)).exp();
        coeff.clamp(0.0, MAX_TIME_CONSTANT_COEFF)
    }}

// ============================================================================
// Window Functions
// ============================================================================

/// Generate Hann window coefficients (allocating version)
pub fn hann_window(size: usize) -> Vec<f32> {
    let mut window = vec![0.0; size];
    hann_window_into(&mut window);
    window
}

/// Generate Hann window coefficients into existing buffer (non-allocating)
pub fn hann_window_into(window: &mut [f32]) {
    let size = window.len();
    if size == 0 {
        return;
    }
    
    let scale = 2.0 * PI / (size - 1) as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let cos_val = (i as f32 * scale).cos();
        *w = 0.5 * (1.0 - cos_val);
    }
}

/// Generate Hamming window coefficients (allocating version)
pub fn hamming_window(size: usize) -> Vec<f32> {
    let mut window = vec![0.0; size];
    hamming_window_into(&mut window);
    window
}

/// Generate Hamming window coefficients into existing buffer (non-allocating)
pub fn hamming_window_into(window: &mut [f32]) {
    let size = window.len();
    if size <= 1 {
        if size == 1 {
            window[0] = 1.0;  // Single sample window
        }
        return;
    }
    
    let scale = 2.0 * PI / (size - 1) as f32;
    
    for (i, w) in window.iter_mut().enumerate() {
        let cos_val = (i as f32 * scale).cos();
        *w = 0.54 - 0.46 * cos_val;
    }
}

/// Apply window function to a buffer in-place
#[inline]
pub fn apply_window(buffer: &mut [f32], window: &[f32]) -> Result<(), &'static str> {
    if buffer.len() != window.len() {
        return Err("Buffer and window size must match");
    }
    
    for (sample, &coeff) in buffer.iter_mut().zip(window.iter()) {
        *sample *= coeff;
    }
    Ok(())
}

/// Apply window function to a buffer in-place (unchecked version for performance)
/// # Safety
/// Caller must ensure buffer.len() == window.len()
#[inline]
pub unsafe fn apply_window_unchecked(buffer: &mut [f32], window: &[f32]) {
    for (sample, &coeff) in buffer.iter_mut().zip(window.iter()) {
        *sample *= coeff;
    }
}

// ============================================================================
// Level Detection
// ============================================================================

/// Calculate RMS (Root Mean Square) of a buffer
#[inline]
pub fn calculate_rms(buffer: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        // SIMD implementation for x86_64 with SSE
        return unsafe { calculate_rms_simd(buffer) };
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse")))]
    {
        calculate_rms_scalar(buffer)
    }
}

/// Scalar implementation of RMS calculation
#[inline]
fn calculate_rms_scalar(buffer: &[f32]) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }
    
    let sum_squares: f32 = buffer.iter().map(|&x| x * x).sum();
    (sum_squares / buffer.len() as f32).sqrt()
}

/// SIMD implementation of RMS calculation for x86_64
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline]
unsafe fn calculate_rms_simd(buffer: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    
    if buffer.is_empty() {
        return 0.0;
    }
    
    let mut sum = _mm_setzero_ps();
    let chunks = buffer.chunks_exact(4);
    let remainder = chunks.remainder();
    
    // Process 4 samples at a time with SSE
    for chunk in chunks {
        let vals = _mm_loadu_ps(chunk.as_ptr());
        let squared = _mm_mul_ps(vals, vals);
        sum = _mm_add_ps(sum, squared);
    }
    
    // Sum the 4 values in the SSE register
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result[0] + result[1] + result[2] + result[3];
    
    // Process remaining samples
    for &sample in remainder {
        total += sample * sample;
    }
    
    (total / buffer.len() as f32).sqrt()
}

/// Calculate peak amplitude of a buffer
#[inline]
pub fn calculate_peak(buffer: &[f32]) -> f32 {
    buffer.iter()
        .map(|&x| x.abs())
        .fold(0.0_f32, f32::max)
}

// ============================================================================
// Parameter Smoothing
// ============================================================================

/// Smooth parameter changes to avoid clicks
/// Uses exponential smoothing with a fast time constant
/// 
/// # Thread Safety
/// This type is not thread-safe. Each thread should have its own instance.
pub struct ParameterSmoother {
    target: f32,
    current: f32,
    coeff: f32,
}

impl ParameterSmoother {
    /// Create a new parameter smoother
    /// smoothing_ms: Smoothing time in milliseconds (typically 5-20ms)
    pub fn new(initial_value: f32, smoothing_ms: f32, sample_rate: f32) -> Self {
        Self {
            target: initial_value,
            current: initial_value,
            coeff: calculate_time_constant(smoothing_ms, sample_rate),
        }
    }

    /// Set the target value
    #[inline]
    pub fn set_target(&mut self, value: f32) {
        self.target = value;
    }

    /// Get next smoothed value
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.current = self.target + self.coeff * (self.current - self.target);
        self.current + DENORMAL_PREVENTION // Prevent denormals
    }

    /// Check if smoothing is complete (within epsilon)
    #[inline]
    pub fn is_settled(&self, epsilon: f32) -> bool {
        (self.current - self.target).abs() < epsilon
    }

    /// Force immediate value change (bypass smoothing)
    pub fn reset(&mut self, value: f32) {
        self.target = value;
        self.current = value;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Clamp a value between min and max
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Linear interpolation between two values
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Mix two signals with a crossfade parameter
/// mix = 0.0: 100% dry, mix = 1.0: 100% wet
#[inline]
pub fn mix_signals(dry: f32, wet: f32, mix: f32) -> f32 {
    let mix_clamped = clamp(mix, 0.0, 1.0);
    dry * (1.0 - mix_clamped) + wet * mix_clamped
}

/// Add small noise to prevent denormal numbers
/// Critical for feedback loops and filters
#[inline]
pub fn denormal_prevention(x: f32) -> f32 {
    x + DENORMAL_PREVENTION
}

/// Fast approximation of sin for LFO generation
/// Accurate to within 0.001 for range [0, 2π]
#[inline]
pub fn fast_sin(phase: f32) -> f32 {
    let x = phase.rem_euclid(2.0 * PI);
    
    // Bhaskara I's sine approximation
    if x < PI {
        (16.0 * x * (PI - x)) / (5.0 * PI * PI - 4.0 * x * (PI - x))
    } else {
        let x = x - PI;
        -((16.0 * x * (PI - x)) / (5.0 * PI * PI - 4.0 * x * (PI - x)))
    }
}

/// Fast approximation of tanh for soft clipping
/// Uses polynomial approximation
#[inline]
pub fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    
    if x.abs() < 3.0 {
        x - (x3 / 3.0) + (2.0 * x5 / 15.0)
    } else {
        x.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_conversion() {
        assert_eq!(linear_to_db(1.0), 0.0);
        assert!((linear_to_db(0.1) - (-20.0)).abs() < 0.001);
        assert_eq!(db_to_linear(0.0), 1.0);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 0.001);
        assert_eq!(linear_to_db(0.0), -f32::INFINITY);
        assert_eq!(db_to_linear(-200.0), 0.0);
    }

    #[test]
    fn test_freq_to_midi_conversion() {
        assert!((freq_to_midi(440.0) - 69.0).abs() < 0.001);
        assert!((freq_to_midi(880.0) - 81.0).abs() < 0.001);
        assert!((midi_to_freq(69.0) - 440.0).abs() < 0.001);
        assert!((midi_to_freq(60.0) - 261.626).abs() < 0.1);
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);

        assert!((window[0] - 0.0).abs() < 0.001);
        assert!((window[1] - 0.75).abs() < 0.001);
        assert!((window[2] - 0.75).abs() < 0.001);
        assert!((window[3] - 0.0).abs() < 0.001);

        // Test non-allocating version
        let mut window2 = vec![0.0; 4];
        hann_window_into(&mut window2);
        assert_eq!(window, window2);
    }

    #[test]
    fn test_hamming_window() {
        let window = hamming_window(4);
        assert_eq!(window.len(), 4);
        assert!(window[0] > 0.0 && window[0] < 0.2);
        
        // Test non-allocating version
        let mut window2 = vec![0.0; 4];
        hamming_window_into(&mut window2);
        assert_eq!(window, window2);
    }
    
    #[test]
    fn test_apply_window() {
        let mut buffer = vec![1.0, 1.0, 1.0, 1.0];
        let window = vec![0.5, 0.75, 0.75, 0.5];
        
        assert!(apply_window(&mut buffer, &window).is_ok());
        assert_eq!(buffer, vec![0.5, 0.75, 0.75, 0.5]);
        
        // Test error case
        let short_window = vec![0.5, 0.5];
        assert!(apply_window(&mut buffer, &short_window).is_err());
    }

    #[test]
    fn test_envelope_follower() {
        let mut env = EnvelopeFollower::new(10.0, 100.0, 48000.0);
        
        // Attack should be faster than release
        let attack_result = env.process(1.0);
        env.reset();
        env.process(1.0);
        env.current_value = 1.0;
        let release_result = env.process(0.0);
        
        assert!(attack_result < 1.0);
        assert!(release_result > 0.0);
        assert!(release_result > (1.0 - attack_result)); // Release is slower
        
        // Test set_times
        env.set_times(5.0, 50.0, 48000.0);
        env.reset();
        let new_attack = env.process(1.0);
        assert!(new_attack > attack_result); // Faster attack
    }

    #[test]
    fn test_rms_calculation() {
        let buffer = vec![0.5, -0.5, 0.5, -0.5];
        assert!((calculate_rms(&buffer) - 0.5).abs() < 0.001);
        
        let silence = vec![0.0; 100];
        assert_eq!(calculate_rms(&silence), 0.0);
    }
    
    #[test]
    fn test_calculate_peak() {
        let buffer = vec![0.5, -0.8, 0.3, -0.2];
        assert!((calculate_peak(&buffer) - 0.8).abs() < 0.001);
        
        let silence = vec![0.0; 100];
        assert_eq!(calculate_peak(&silence), 0.0);
    }

    #[test]
    fn test_parameter_smoother() {
        let mut smoother = ParameterSmoother::new(0.0, 10.0, 48000.0);
        smoother.set_target(1.0);
        
        let val1 = smoother.next();
        let val2 = smoother.next();
        
        assert!(val1 > 0.0 && val1 < 1.0);
        assert!(val2 > val1 && val2 < 1.0);
        assert!(!smoother.is_settled(0.001));
        
        // Eventually settles
        for _ in 0..10000 {
            smoother.next();
        }
        assert!(smoother.is_settled(0.001));
    }
    
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(clamp(-0.5, 0.0, 1.0), 0.0);
        assert_eq!(clamp(1.5, 0.0, 1.0), 1.0);
    }
    
    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < 0.001);
        assert!((lerp(-1.0, 1.0, 0.25) - (-0.5)).abs() < 0.001);
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }
    
    #[test]
    fn test_mix_signals() {
        assert_eq!(mix_signals(1.0, 0.0, 0.0), 1.0);
        assert_eq!(mix_signals(1.0, 0.0, 1.0), 0.0);
        assert!((mix_signals(1.0, 0.5, 0.5) - 0.75).abs() < 0.001);
        
        // Test clamping
        assert_eq!(mix_signals(1.0, 0.0, -0.5), 1.0);
        assert_eq!(mix_signals(1.0, 0.0, 1.5), 0.0);
    }
    
    #[test]
    fn test_fast_sin() {
        // Test accuracy at key points
        assert!((fast_sin(0.0) - 0.0).abs() < 0.01);
        assert!((fast_sin(PI / 2.0) - 1.0).abs() < 0.01);
        assert!((fast_sin(PI) - 0.0).abs() < 0.01);
        assert!((fast_sin(3.0 * PI / 2.0) - (-1.0)).abs() < 0.01);
        
        // Test wraparound
        assert!((fast_sin(2.0 * PI) - 0.0).abs() < 0.01);
        assert!((fast_sin(2.5 * PI) - fast_sin(0.5 * PI)).abs() < 0.01);
    }
    
    #[test]
    fn test_fast_tanh() {
        assert!((fast_tanh(0.0) - 0.0).abs() < 0.001);
        assert!(fast_tanh(1.0) > 0.7 && fast_tanh(1.0) < 0.8);
        assert!((fast_tanh(5.0) - 1.0).abs() < 0.001);
        assert!((fast_tanh(-5.0) - (-1.0)).abs() < 0.001);
    }
}
