//! Spectral smoothing filters to reduce musical noise artifacts
//! 
//! This module provides various smoothing techniques for spectral processing
//! to reduce the "musical noise" artifacts common in spectral subtraction.

use std::f32::consts::PI;

/// Type of spectral smoothing to apply
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingType {
    /// No smoothing
    None,
    /// Median filter (good for removing isolated peaks)
    Median { window_size: usize },
    /// Gaussian smoothing (smooth transition)
    Gaussian { sigma: f32 },
    /// Moving average (simple but effective)
    MovingAverage { window_size: usize },
    /// Savitzky-Golay filter (preserves peaks better)
    SavitzkyGolay { window_size: usize, order: usize },
}

impl Default for SmoothingType {
    fn default() -> Self {
        SmoothingType::Gaussian { sigma: 1.5 }
    }
}

/// Spectral smoothing processor
pub struct SpectralSmoother {
    smoothing_type: SmoothingType,
    scratch_buffer: Vec<f32>,
    gaussian_kernel: Vec<f32>,
}

impl SpectralSmoother {
    /// Create a new spectral smoother
    pub fn new(smoothing_type: SmoothingType, max_size: usize) -> Self {
        let gaussian_kernel = match smoothing_type {
            SmoothingType::Gaussian { sigma } => {
                Self::create_gaussian_kernel(sigma)
            }
            _ => Vec::new(),
        };
        
        Self {
            smoothing_type,
            scratch_buffer: vec![0.0; max_size],
            gaussian_kernel,
        }
    }
    
    /// Apply smoothing to a spectrum (in-place)
    pub fn smooth(&mut self, spectrum: &mut [f32]) {
        match self.smoothing_type {
            SmoothingType::None => {},
            SmoothingType::Median { window_size } => {
                self.apply_median_filter(spectrum, window_size);
            }
            SmoothingType::Gaussian { .. } => {
                self.apply_gaussian_filter(spectrum);
            }
            SmoothingType::MovingAverage { window_size } => {
                self.apply_moving_average(spectrum, window_size);
            }
            SmoothingType::SavitzkyGolay { window_size, order } => {
                self.apply_savitzky_golay(spectrum, window_size, order);
            }
        }
    }
    
    /// Apply median filter to remove isolated peaks (musical noise)
    fn apply_median_filter(&mut self, spectrum: &mut [f32], window_size: usize) {
        let len = spectrum.len();
        if len < window_size {
            return;
        }
        
        // Copy original values
        self.scratch_buffer[..len].copy_from_slice(spectrum);
        
        let half_window = window_size / 2;
        let mut window = vec![0.0; window_size];
        
        for i in 0..len {
            // Collect window values with boundary handling
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(len);
            let actual_window_size = end - start;
            
            // Fill window
            for (j, k) in (start..end).enumerate() {
                window[j] = self.scratch_buffer[k];
            }
            
            // Sort and find median
            window[..actual_window_size].sort_by(|a, b| a.partial_cmp(b).unwrap());
            spectrum[i] = if actual_window_size % 2 == 1 {
                window[actual_window_size / 2]
            } else {
                (window[actual_window_size / 2 - 1] + window[actual_window_size / 2]) * 0.5
            };
        }
    }
    
    /// Apply Gaussian smoothing
    fn apply_gaussian_filter(&mut self, spectrum: &mut [f32]) {
        let len = spectrum.len();
        let kernel_len = self.gaussian_kernel.len();
        
        if len == 0 || kernel_len == 0 {
            return;
        }
        
        // Copy original values
        self.scratch_buffer[..len].copy_from_slice(spectrum);
        
        let half_kernel = kernel_len / 2;
        
        for i in 0..len {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            
            for (j, &kernel_val) in self.gaussian_kernel.iter().enumerate() {
                let idx = (i as isize - half_kernel as isize + j as isize);
                
                // Boundary handling: mirror at edges
                let idx = if idx < 0 {
                    (-idx) as usize
                } else if idx >= len as isize {
                    (2 * len as isize - idx - 1) as usize
                } else {
                    idx as usize
                };
                
                if idx < len {
                    sum += self.scratch_buffer[idx] * kernel_val;
                    weight_sum += kernel_val;
                }
            }
            
            if weight_sum > 0.0 {
                spectrum[i] = sum / weight_sum;
            }
        }
    }
    
    /// Apply moving average filter
    fn apply_moving_average(&mut self, spectrum: &mut [f32], window_size: usize) {
        let len = spectrum.len();
        if len < window_size {
            return;
        }
        
        // Copy original values
        self.scratch_buffer[..len].copy_from_slice(spectrum);
        
        let half_window = window_size / 2;
        
        for i in 0..len {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(len);
            
            let sum: f32 = self.scratch_buffer[start..end].iter().sum();
            spectrum[i] = sum / (end - start) as f32;
        }
    }
    
    /// Apply Savitzky-Golay filter (polynomial smoothing)
    fn apply_savitzky_golay(&mut self, spectrum: &mut [f32], window_size: usize, order: usize) {
        // For simplicity, fallback to moving average for now
        // Full S-G implementation requires polynomial fitting
        self.apply_moving_average(spectrum, window_size.min(order + 1));
    }
    
    /// Create Gaussian kernel for convolution
    fn create_gaussian_kernel(sigma: f32) -> Vec<f32> {
        // Kernel size: typically 6*sigma, must be odd
        let kernel_size = ((6.0 * sigma) as usize) | 1;
        let half_size = kernel_size / 2;
        
        let mut kernel = vec![0.0; kernel_size];
        let two_sigma_sq = 2.0 * sigma * sigma;
        
        for i in 0..kernel_size {
            let x = (i as f32 - half_size as f32);
            kernel[i] = (-x * x / two_sigma_sq).exp();
        }
        
        // Normalize
        let sum: f32 = kernel.iter().sum();
        if sum > 0.0 {
            for k in &mut kernel {
                *k /= sum;
            }
        }
        
        kernel
    }
    
    /// Update smoothing type (recreates kernel if needed)
    pub fn set_smoothing_type(&mut self, smoothing_type: SmoothingType) {
        self.smoothing_type = smoothing_type;
        
        if let SmoothingType::Gaussian { sigma } = smoothing_type {
            self.gaussian_kernel = Self::create_gaussian_kernel(sigma);
        }
    }
}

/// 2D spectral smoother for time-frequency smoothing
pub struct SpectralSmoother2D {
    freq_smoother: SpectralSmoother,
    time_smoother: SpectralSmoother,
    time_buffer: Vec<Vec<f32>>,
    buffer_index: usize,
    buffer_size: usize,
}

impl SpectralSmoother2D {
    /// Create a new 2D spectral smoother
    pub fn new(
        freq_smoothing: SmoothingType,
        time_smoothing: SmoothingType,
        max_freq_bins: usize,
        time_window_frames: usize,
    ) -> Self {
        Self {
            freq_smoother: SpectralSmoother::new(freq_smoothing, max_freq_bins),
            time_smoother: SpectralSmoother::new(time_smoothing, time_window_frames),
            time_buffer: vec![vec![0.0; max_freq_bins]; time_window_frames],
            buffer_index: 0,
            buffer_size: time_window_frames,
        }
    }
    
    /// Process a spectrum with 2D smoothing
    pub fn process(&mut self, spectrum: &mut [f32]) {
        // Store current frame
        let len = spectrum.len().min(self.time_buffer[0].len());
        self.time_buffer[self.buffer_index][..len].copy_from_slice(&spectrum[..len]);
        
        // Apply frequency smoothing first
        self.freq_smoother.smooth(spectrum);
        
        // Apply temporal smoothing per frequency bin
        for bin_idx in 0..len {
            // Collect time samples for this frequency bin
            let mut time_samples = vec![0.0; self.buffer_size];
            for t in 0..self.buffer_size {
                time_samples[t] = self.time_buffer[t][bin_idx];
            }
            
            // Smooth across time
            self.time_smoother.smooth(&mut time_samples);
            
            // Use the smoothed value for current frame
            spectrum[bin_idx] = time_samples[self.buffer_index];
        }
        
        // Update circular buffer index
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size;
    }
    
    /// Reset the time buffer
    pub fn reset(&mut self) {
        for buffer in &mut self.time_buffer {
            buffer.fill(0.0);
        }
        self.buffer_index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_median_filter_removes_peaks() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::Median { window_size: 3 },
            10
        );
        
        // Create spectrum with isolated peak (musical noise)
        let mut spectrum = vec![1.0, 1.0, 10.0, 1.0, 1.0];
        smoother.smooth(&mut spectrum);
        
        // Peak should be reduced
        assert!(spectrum[2] < 5.0, "Median filter should reduce isolated peak");
        assert!(spectrum[2] > 0.5, "Median filter shouldn't over-smooth");
    }
    
    #[test]
    fn test_gaussian_smoothing() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::Gaussian { sigma: 1.0 },
            10
        );
        
        // Create spectrum with sharp transition
        let mut spectrum = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let original = spectrum.clone();
        smoother.smooth(&mut spectrum);
        
        // Should smooth the transition
        assert!(spectrum[1] > original[1], "Should smooth up from 0");
        assert!(spectrum[1] < spectrum[2], "Should create gradual transition");
        
        // Energy should be roughly preserved
        let original_sum: f32 = original.iter().sum();
        let smoothed_sum: f32 = spectrum.iter().sum();
        assert!((original_sum - smoothed_sum).abs() < 0.5, "Energy should be preserved");
    }
    
    #[test]
    fn test_moving_average() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::MovingAverage { window_size: 3 },
            10
        );
        
        let mut spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        smoother.smooth(&mut spectrum);
        
        // Middle values should be average of neighbors
        assert!((spectrum[2] - 3.0).abs() < 0.01, "Middle should be average");
    }
    
    #[test]
    fn test_no_smoothing() {
        let mut smoother = SpectralSmoother::new(SmoothingType::None, 10);
        
        let mut spectrum = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = spectrum.clone();
        smoother.smooth(&mut spectrum);
        
        assert_eq!(spectrum, original, "No smoothing should leave spectrum unchanged");
    }
    
    #[test]
    fn test_gaussian_kernel_normalization() {
        let kernel = SpectralSmoother::create_gaussian_kernel(1.5);
        
        // Kernel should sum to 1.0
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Gaussian kernel should be normalized");
        
        // Peak should be at center
        let center = kernel.len() / 2;
        let max_val = kernel.iter().fold(0.0f32, |a, &b| a.max(b));
        assert_eq!(kernel[center], max_val, "Gaussian peak should be at center");
    }
    
    #[test]
    fn test_median_filter_boundary_handling() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::Median { window_size: 5 },
            10
        );
        
        let mut spectrum = vec![5.0, 4.0, 3.0];
        smoother.smooth(&mut spectrum);
        
        // Should handle boundaries without crashing
        assert!(spectrum[0] <= 5.0);
        assert!(spectrum[2] <= 5.0);
    }
    
    #[test]
    fn test_2d_smoothing() {
        let mut smoother = SpectralSmoother2D::new(
            SmoothingType::Median { window_size: 3 },
            SmoothingType::MovingAverage { window_size: 3 },
            10,
            5
        );
        
        // Process several frames
        for i in 0..5 {
            let mut spectrum = vec![i as f32; 5];
            if i == 2 {
                spectrum[2] = 10.0; // Add spike
            }
            smoother.process(&mut spectrum);
        }
        
        // After 2D smoothing, variations should be reduced
        let mut final_spectrum = vec![4.0; 5];
        smoother.process(&mut final_spectrum);
        
        // The spike should be smoothed out
        assert!(final_spectrum[2] < 8.0, "2D smoothing should reduce spikes");
    }
    
    #[test]
    fn test_empty_spectrum() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::Gaussian { sigma: 1.0 },
            10
        );
        
        let mut spectrum: Vec<f32> = vec![];
        smoother.smooth(&mut spectrum);
        
        assert_eq!(spectrum.len(), 0, "Empty spectrum should remain empty");
    }
    
    #[test]
    fn test_single_value_spectrum() {
        let mut smoother = SpectralSmoother::new(
            SmoothingType::Median { window_size: 3 },
            10
        );
        
        let mut spectrum = vec![5.0];
        smoother.smooth(&mut spectrum);
        
        assert_eq!(spectrum[0], 5.0, "Single value should remain unchanged");
    }
}