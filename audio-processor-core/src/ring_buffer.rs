//! Lock-free ring buffer for real-time audio processing
//! 
//! This module provides a high-performance, cache-optimized ring buffer
//! specifically designed for single-producer single-consumer scenarios
//! common in audio processing pipelines.
//! 
//! # Testing
//! 
//! Run the comprehensive test suite with:
//! ```bash
//! cargo test
//! ```
//! 
//! Run with Miri to check for undefined behavior:
//! ```bash
//! cargo +nightly miri test
//! ```
//! 
//! Run with ThreadSanitizer to check for data races:
//! ```bash
//! RUSTFLAGS="-Z sanitizer=thread" cargo test --target x86_64-unknown-linux-gnu
//! ```
//! 
//! Run benchmarks:
//! ```bash
//! cargo test --release bench_throughput -- --nocapture
//! ```

use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::cell::UnsafeCell;
use std::ptr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Standard x86_64 cache line size in bytes
const CACHE_LINE_SIZE: usize = 64;

/// Cache-line aligned wrapper to prevent false sharing
#[repr(align(64))]
struct CacheAligned<T>(T);

/// Error types for ring buffer operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RingBufferError {
    BufferFull,
    BufferEmpty,
    InsufficientSpace { requested: usize, available: usize },
    InvalidSize,
}

/// Statistics for monitoring ring buffer performance
#[derive(Debug, Default)]
pub struct RingBufferStats {
    pub total_written: AtomicU64,
    pub total_read: AtomicU64,
    pub overflow_count: AtomicU64,
    pub underflow_count: AtomicU64,
}

impl RingBufferStats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn reset(&self) {
        self.total_written.store(0, Ordering::Relaxed);
        self.total_read.store(0, Ordering::Relaxed);
        self.overflow_count.store(0, Ordering::Relaxed);
        self.underflow_count.store(0, Ordering::Relaxed);
    }
}

/// A lock-free, cache-optimized ring buffer for single-producer single-consumer scenarios
/// 
/// # Memory Ordering Strategy
/// - Producer uses Relaxed for write_pos (it owns this)
/// - Producer uses Acquire for read_pos (to see consumer progress)
/// - Consumer uses Relaxed for read_pos (it owns this)
/// - Consumer uses Acquire for write_pos (to see producer data)
/// - Both use Release on updates to ensure visibility
/// 
/// # Thread Safety
/// This is safe for exactly one producer and one consumer thread.
/// The producer owns write operations, the consumer owns read operations.
/// 
/// # Example
/// ```
/// use ring_buffer::RingBuffer;
/// use std::thread;
/// 
/// let buffer = RingBuffer::new(1024).unwrap();
/// let buffer_clone = buffer.clone(); // Arc-wrapped internally
/// 
/// // Producer thread
/// thread::spawn(move || {
///     let data = vec![1.0; 128];
///     buffer.write(&data).unwrap();
/// });
/// 
/// // Consumer thread
/// thread::spawn(move || {
///     let mut output = vec![0.0; 128];
///     buffer_clone.read(&mut output).unwrap();
/// });
/// ```
pub struct RingBuffer {
    /// Internal buffer storage - using UnsafeCell for interior mutability
    buffer: Box<[UnsafeCell<f32>]>,
    /// Size of the buffer (always power of 2)
    size: usize,
    /// Mask for efficient modulo operation (size - 1)
    mask: usize,
    
    // Cache-line separated positions to avoid false sharing
    /// Write position (owned by producer)
    write_pos: CacheAligned<AtomicUsize>,
    /// Padding to separate cache lines
    _padding: [u8; CACHE_LINE_SIZE],
    /// Read position (owned by consumer)
    read_pos: CacheAligned<AtomicUsize>,
    
    /// Optional statistics tracking
    pub stats: RingBufferStats,
}

// Mark as Send + Sync since we handle synchronization
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Create a new ring buffer with specified size
    /// Size will be rounded up to nearest power of 2
    /// Returns error if size is 0 or would overflow
    pub fn new(requested_size: usize) -> Result<Self, RingBufferError> {
        if requested_size == 0 {
            return Err(RingBufferError::InvalidSize);
        }
        
        // Round up to power of 2 for efficient wrapping
        let size = requested_size
            .checked_next_power_of_two()
            .ok_or(RingBufferError::InvalidSize)?;
        let mask = size - 1;
        
        // Create buffer with UnsafeCell for interior mutability
        let buffer: Box<[UnsafeCell<f32>]> = (0..size)
            .map(|_| UnsafeCell::new(0.0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        
        Ok(Self {
            buffer,
            size,
            mask,
            write_pos: CacheAligned(AtomicUsize::new(0)),
            _padding: [0u8; CACHE_LINE_SIZE],
            read_pos: CacheAligned(AtomicUsize::new(0)),
            stats: RingBufferStats::new(),
        })
    }
    
    /// Get the actual size of the buffer
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.size
    }
    
    /// Get number of samples available for reading
    /// This is safe to call from the consumer thread
    #[inline]
    pub fn available(&self) -> usize {
        let write = self.write_pos.0.load(Ordering::Acquire);
        let read = self.read_pos.0.load(Ordering::Relaxed); // Consumer owns read_pos
        write.wrapping_sub(read)
    }
    
    /// Get number of samples available for writing
    /// This is safe to call from the producer thread
    #[inline]
    pub fn space(&self) -> usize {
        let write = self.write_pos.0.load(Ordering::Relaxed); // Producer owns write_pos
        let read = self.read_pos.0.load(Ordering::Acquire);
        self.size.saturating_sub(write.wrapping_sub(read)).saturating_sub(1)
    }
    
    /// Check if buffer is empty (consumer perspective)
    #[inline]
    pub fn is_empty(&self) -> bool {
        let write = self.write_pos.0.load(Ordering::Acquire);
        let read = self.read_pos.0.load(Ordering::Relaxed);
        write == read
    }
    
    pub fn reset(&mut self) {
        self.write_pos.0.store(0, Ordering::Relaxed);
        self.read_pos.0.store(0, Ordering::Relaxed);

        // Zero the buffer
        let buffer_ptr = self.buffer.as_ptr() as *mut f32;
        unsafe {
            ptr::write_bytes(buffer_ptr, 0, self.size);
        }

        self.stats.reset();
    }

    // ========================================================================
    // Producer Operations (call from producer thread only)
    // ========================================================================
    
    /// Write a frame of samples (producer only)
    /// Returns Ok(samples_written) or error
    pub fn write(&self, frame: &[f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);  // Empty write is a no-op
        }

        if frame.len() > self.size {
            return Err(RingBufferError::InvalidSize);  // Frame too large
        }

        let write_pos = self.write_pos.0.load(Ordering::Relaxed);
        let read_pos = self.read_pos.0.load(Ordering::Acquire);
        
        let available_space = self.size.saturating_sub(write_pos.wrapping_sub(read_pos)).saturating_sub(1);
        
        if available_space == 0 {
            self.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::BufferFull);
        }
        
        let to_write = frame.len().min(available_space);
        
        if to_write < frame.len() {
            self.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available: available_space,
            });
        }
        
        // Write data using optimized copy
        unsafe {
            self.write_unchecked(write_pos, &frame[..to_write]);
        }
        
        // Update write position with Release to ensure data is visible
        self.write_pos.0.store(write_pos.wrapping_add(to_write), Ordering::Release);
        self.stats.total_written.fetch_add(to_write as u64, Ordering::Relaxed);
        
        Ok(to_write)
    }
    
    /// Write samples, partially if necessary (producer only)
    pub fn write_available(&self, frame: &[f32]) -> usize {
        if frame.is_empty() {
            return 0;
        }
        
        let write_pos = self.write_pos.0.load(Ordering::Relaxed);
        let read_pos = self.read_pos.0.load(Ordering::Acquire);
        
        let available_space = self.size.saturating_sub(write_pos.wrapping_sub(read_pos)).saturating_sub(1);
        let to_write = frame.len().min(available_space);
        
        if to_write == 0 {
            self.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return 0;
        }
        
        unsafe {
            self.write_unchecked(write_pos, &frame[..to_write]);
        }
        
        self.write_pos.0.store(write_pos.wrapping_add(to_write), Ordering::Release);
        self.stats.total_written.fetch_add(to_write as u64, Ordering::Relaxed);
        to_write
    }
    
    /// Internal unchecked write using optimized memory copy
    #[inline]
    unsafe fn write_unchecked(&self, start_pos: usize, data: &[f32]) {
        let len = data.len();
        let start_idx = start_pos & self.mask;
        let first_part_len = (self.size - start_idx).min(len);
        
        // Get raw pointer for direct memory operations
        let buffer_ptr = self.buffer.as_ptr() as *mut f32;
        
        // Use ptr::copy_nonoverlapping for better vectorization
        ptr::copy_nonoverlapping(
            data.as_ptr(),
            buffer_ptr.add(start_idx),
            first_part_len
        );
        
        // Copy wraparound part if necessary
        if len > first_part_len {
            ptr::copy_nonoverlapping(
                data.as_ptr().add(first_part_len),
                buffer_ptr,
                len - first_part_len
            );
        }
    }
    
    // ========================================================================
    // Consumer Operations (call from consumer thread only)
    // ========================================================================
    
    /// Read a frame of samples (consumer only)
    /// Returns Ok(samples_read) or error
    pub fn read(&self, frame: &mut [f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);  // Empty write is a no-op
        }
                
        let read_pos = self.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        
        if available == 0 {
            self.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::BufferEmpty);
        }
        
        let to_read = frame.len().min(available);
        
        if to_read < frame.len() {
            self.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available,
            });
        }
        
        // Read data using optimized copy
        unsafe {
            self.read_unchecked(read_pos, &mut frame[..to_read]);
        }
        
        // Update read position using fetch_add for clarity
        self.read_pos.0.fetch_add(to_read, Ordering::Release);
        self.stats.total_read.fetch_add(to_read as u64, Ordering::Relaxed);
        
        Ok(to_read)
    }
    
    /// Read available samples, fill rest with zeros (consumer only)
    pub fn read_available(&self, frame: &mut [f32]) -> usize {
        if frame.is_empty() {
            return 0;
        }
       
        let read_pos = self.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        let to_read = frame.len().min(available);
        
        if to_read == 0 {
            frame.fill(0.0);
            self.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return 0;
        }
        
        unsafe {
            self.read_unchecked(read_pos, &mut frame[..to_read]);
        }
        
        // Fill rest with silence
        if to_read < frame.len() {
            frame[to_read..].fill(0.0);
        }
        
        self.read_pos.0.fetch_add(to_read, Ordering::Release);
        self.stats.total_read.fetch_add(to_read as u64, Ordering::Relaxed);
        to_read
    }
    
    /// Internal unchecked read using optimized memory copy
    #[inline]
    unsafe fn read_unchecked(&self, start_pos: usize, data: &mut [f32]) {
        let len = data.len();
        let start_idx = start_pos & self.mask;
        let first_part_len = (self.size - start_idx).min(len);
        
        // Get raw pointer for direct memory operations
        let buffer_ptr = self.buffer.as_ptr() as *const f32;
        
        // Use ptr::copy_nonoverlapping for better vectorization
        ptr::copy_nonoverlapping(
            buffer_ptr.add(start_idx),
            data.as_mut_ptr(),
            first_part_len
        );
        
        // Copy wraparound part if necessary
        if len > first_part_len {
            ptr::copy_nonoverlapping(
                buffer_ptr,
                data.as_mut_ptr().add(first_part_len),
                len - first_part_len
            );
        }
    }
    
    /// Peek at samples without consuming them (consumer only)
    pub fn peek(&self, output: &mut [f32]) -> usize {
        let read_pos = self.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        let to_read = output.len().min(available);
        
        if to_read == 0 {
            output.fill(0.0);
            return 0;
        }
        
        unsafe {
            self.read_unchecked(read_pos, &mut output[..to_read]);
        }
        
        if to_read < output.len() {
            output[to_read..].fill(0.0);
        }
        
        to_read
    }
    
    /// Skip samples without reading them (consumer only)
    pub fn skip(&self, count: usize) -> usize {
        let read_pos = self.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        let to_skip = count.min(available);
        
        if to_skip > 0 {
            // Use fetch_add for clarity and potential optimization
            self.read_pos.0.fetch_add(to_skip, Ordering::Release);
            self.stats.total_read.fetch_add(to_skip as u64, Ordering::Relaxed);
        }
        
        to_skip
    }
}

/// Static-sized ring buffer for compile-time known sizes
/// Provides better optimization opportunities
pub struct StaticRingBuffer<const SIZE: usize> {
    buffer: Box<[UnsafeCell<f32>; SIZE]>,
    write_pos: CacheAligned<AtomicUsize>,
    _padding: [u8; CACHE_LINE_SIZE],
    read_pos: CacheAligned<AtomicUsize>,
    pub stats: RingBufferStats,
}

unsafe impl<const SIZE: usize> Send for StaticRingBuffer<SIZE> {}
unsafe impl<const SIZE: usize> Sync for StaticRingBuffer<SIZE> {}

impl<const SIZE: usize> StaticRingBuffer<SIZE> {
    const MASK: usize = SIZE - 1;
    
    /// Create a new static ring buffer
    /// Size must be a power of 2 (checked at compile time)
    pub fn new() -> Self {
        const { assert!(SIZE.is_power_of_two(), "SIZE must be a power of 2") };
        const { assert!(SIZE > 0, "SIZE must be greater than 0") };
        
        // Create array with UnsafeCell
        let buffer = Box::new([(); SIZE].map(|_| UnsafeCell::new(0.0f32)));
        
        Self {
            buffer,
            write_pos: CacheAligned(AtomicUsize::new(0)),
            _padding: [0u8; CACHE_LINE_SIZE],
            read_pos: CacheAligned(AtomicUsize::new(0)),
            stats: RingBufferStats::new(),
        }
    }
    
    #[inline]
    pub const fn capacity(&self) -> usize {
        SIZE
    }
    
    #[inline]
    pub fn available(&self) -> usize {
        let write = self.write_pos.0.load(Ordering::Acquire);
        let read = self.read_pos.0.load(Ordering::Relaxed);
        write.wrapping_sub(read)
    }
    
    #[inline]
    pub fn space(&self) -> usize {
        let write = self.write_pos.0.load(Ordering::Relaxed);
        let read = self.read_pos.0.load(Ordering::Acquire);
        SIZE.saturating_sub(write.wrapping_sub(read)).saturating_sub(1)
    }
    
    /// Write a frame of samples (producer only)
    pub fn write(&self, frame: &[f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);  // Empty write is a no-op
        }
                        
        let write_pos = self.write_pos.0.load(Ordering::Relaxed);
        let read_pos = self.read_pos.0.load(Ordering::Acquire);
        
        let available_space = SIZE.saturating_sub(write_pos.wrapping_sub(read_pos)).saturating_sub(1);
        
        if available_space < frame.len() {
            self.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available: available_space,
            });
        }
        
        unsafe {
            self.write_unchecked(write_pos, frame);
        }
        
        self.write_pos.0.store(write_pos.wrapping_add(frame.len()), Ordering::Release);
        self.stats.total_written.fetch_add(frame.len() as u64, Ordering::Relaxed);
        Ok(frame.len())
    }
    
    /// Internal unchecked write with compile-time optimizations
    #[inline]
    unsafe fn write_unchecked(&self, start_pos: usize, data: &[f32]) {
        let len = data.len();
        let start_idx = start_pos & Self::MASK;
        let first_part_len = (SIZE - start_idx).min(len);
        
        let buffer_ptr = self.buffer.as_ptr() as *mut f32;
        
        ptr::copy_nonoverlapping(
            data.as_ptr(),
            buffer_ptr.add(start_idx),
            first_part_len
        );
        
        if len > first_part_len {
            ptr::copy_nonoverlapping(
                data.as_ptr().add(first_part_len),
                buffer_ptr,
                len - first_part_len
            );
        }
    }
    
    /// Read a frame of samples (consumer only)
    pub fn read(&self, frame: &mut [f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);  // Empty write is a no-op
        }
                
        let read_pos = self.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        
        if available < frame.len() {
            self.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available,
            });
        }
        
        unsafe {
            self.read_unchecked(read_pos, frame);
        }
        
        self.read_pos.0.fetch_add(frame.len(), Ordering::Release);
        self.stats.total_read.fetch_add(frame.len() as u64, Ordering::Relaxed);
        Ok(frame.len())
    }
    
    /// Internal unchecked read with compile-time optimizations
    #[inline]
    unsafe fn read_unchecked(&self, start_pos: usize, data: &mut [f32]) {
        let len = data.len();
        let start_idx = start_pos & Self::MASK;
        let first_part_len = (SIZE - start_idx).min(len);
        
        let buffer_ptr = self.buffer.as_ptr() as *const f32;
        
        ptr::copy_nonoverlapping(
            buffer_ptr.add(start_idx),
            data.as_mut_ptr(),
            first_part_len
        );
        
        if len > first_part_len {
            ptr::copy_nonoverlapping(
                buffer_ptr,
                data.as_mut_ptr().add(first_part_len),
                len - first_part_len
            );
        }
    }
}

impl<const SIZE: usize> Default for StaticRingBuffer<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Double buffer for smooth audio processing with atomic swapping
/// 
/// # Safety
/// The `active_mut` method requires careful synchronization.
/// No swaps should occur while holding a mutable reference.
pub struct DoubleBuffer {
    buffers: [Vec<f32>; 2],
    active_buffer: AtomicUsize,
    size: usize,
}

impl DoubleBuffer {
    /// Create a new double buffer with specified size
    pub fn new(size: usize) -> Self {
        Self {
            buffers: [vec![0.0; size], vec![0.0; size]],
            active_buffer: AtomicUsize::new(0),
            size,
        }
    }
    
    /// Get the active buffer index
    #[inline]
    pub fn active_index(&self) -> usize {
        self.active_buffer.load(Ordering::Acquire)
    }
    
    /// Get the active buffer for reading
    #[inline]
    pub fn active(&self) -> &[f32] {
        &self.buffers[self.active_index()]
    }
    
    /// Get the active buffer for writing
    /// 
    /// # Safety
    /// Caller must ensure:
    /// - No concurrent access to the same buffer
    /// - No swap() calls while holding this reference
    /// - Reference is dropped before next swap()
    #[inline]
    pub unsafe fn active_mut(&mut self) -> &mut [f32] {
        let idx = self.active_index();
        &mut self.buffers[idx]
    }
    
    /// Get the inactive buffer index
    #[inline]
    pub fn inactive_index(&self) -> usize {
        1 - self.active_index()
    }
    
    /// Get the inactive buffer for reading/processing
    #[inline]
    pub fn inactive(&self) -> &[f32] {
        &self.buffers[self.inactive_index()]
    }
    
    /// Swap active and inactive buffers atomically
    #[inline]
    pub fn swap(&self) {
        self.active_buffer.fetch_xor(1, Ordering::AcqRel);
    }
    
    /// Clear both buffers
    /// 
    /// # Safety
    /// Caller must ensure no concurrent access during clear
    pub unsafe fn clear(&mut self) {
        self.buffers[0].fill(0.0);
        self.buffers[1].fill(0.0);
    }
    
    /// Get buffer size
    #[inline]
    pub const fn size(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_basic() {
        let rb = RingBuffer::new(8).unwrap();
        assert_eq!(rb.capacity(), 8);
        assert_eq!(rb.available(), 0);
        assert_eq!(rb.space(), 7);
        
        let input = vec![1.0, 2.0, 3.0];
        assert_eq!(rb.write(&input).unwrap(), 3);
        assert_eq!(rb.available(), 3);
        
        let mut output = vec![0.0; 3];
        assert_eq!(rb.read(&mut output).unwrap(), 3);
        assert_eq!(output, input);
    }
    
    #[test]
    fn test_stats_tracking() {
        let rb = RingBuffer::new(8).unwrap();
        
        rb.write(&vec![1.0; 4]).unwrap();
        assert_eq!(rb.stats.total_written.load(Ordering::Relaxed), 4);
        
        let mut output = vec![0.0; 4];
        rb.read(&mut output).unwrap();
        assert_eq!(rb.stats.total_read.load(Ordering::Relaxed), 4);
        
        // Test overflow
        rb.write(&vec![1.0; 8]).ok();
        assert_eq!(rb.stats.overflow_count.load(Ordering::Relaxed), 1);
    }
    
    #[cfg(all(test, not(miri)))]
    #[test]
    fn bench_throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        
        let rb = RingBuffer::new(4096).unwrap();
        let data = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 64];
        
        let start = Instant::now();
        let iterations = 1_000_000;
        
        for _ in 0..iterations {
            rb.write_available(black_box(&data));
            rb.read_available(black_box(&mut output));
        }
        
        let elapsed = start.elapsed();
        let throughput = (iterations * 64) as f64 / elapsed.as_secs_f64();
        println!("Throughput: {:.2} samples/sec", throughput);
        
        // Verify stats
        assert_eq!(
            rb.stats.total_written.load(Ordering::Relaxed),
            rb.stats.total_read.load(Ordering::Relaxed)
        );
    }

    // ========================================================================
    // Basic Functionality Tests
    // ========================================================================
    
    #[test]
    fn test_new_and_capacity() {
        // Test power-of-2 rounding
        assert_eq!(RingBuffer::new(10).unwrap().capacity(), 16);
        assert_eq!(RingBuffer::new(32).unwrap().capacity(), 32);
        assert_eq!(RingBuffer::new(33).unwrap().capacity(), 64);
        assert_eq!(RingBuffer::new(1).unwrap().capacity(), 1);

        // Test invalid sizes
        assert!(matches!(RingBuffer::new(0), Err(RingBufferError::InvalidSize)));
    }

    #[test]
    fn test_basic_read_write() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Initial state
        assert_eq!(rb.available(), 0);
        assert_eq!(rb.space(), 7);
        assert!(rb.is_empty());
        
        // Write and verify
        let input = vec![1.0, 2.0, 3.0];
        assert_eq!(rb.write(&input).unwrap(), 3);
        assert_eq!(rb.available(), 3);
        assert_eq!(rb.space(), 4);
        assert!(!rb.is_empty());
        
        // Read and verify
        let mut output = vec![0.0; 3];
        assert_eq!(rb.read(&mut output).unwrap(), 3);
        assert_eq!(output, input);
        assert!(rb.is_empty());
    }
    
    // ========================================================================
    // Wraparound Tests - CRITICAL
    // ========================================================================
    
    #[test]
    fn test_wraparound_basic() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Fill most of buffer
        assert_eq!(rb.write(&vec![1.0; 6]).unwrap(), 6);
        
        // Read some to make space at beginning
        let mut temp = vec![0.0; 4];
        assert_eq!(rb.read(&mut temp).unwrap(), 4);
        assert_eq!(temp, vec![1.0; 4]);
        
        // Write across boundary (2 at end, 3 at beginning)
        assert_eq!(rb.write(&vec![2.0; 5]).unwrap(), 5);
        
        // Read across boundary
        let mut result = vec![0.0; 7];
        assert_eq!(rb.read(&mut result).unwrap(), 7);
        assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }
    
    #[test]
    fn test_wraparound_stress() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Many wraparounds with varying sizes
        for i in 0..100 {
            let write_size = (i % 5) + 1;
            let data = vec![i as f32; write_size];
            
            // Write what we can
            let _written = rb.write_available(&data);
            
            // Read some to make space
            if rb.available() >= 3 {
                let mut temp = vec![0.0; 3];
                rb.read(&mut temp).unwrap();
            }
        }
        
        // Verify buffer still functional
        assert!(rb.available() <= 7);
        assert!(rb.space() + rb.available() == 7);
    }
    
    // ========================================================================
    // Boundary Condition Tests
    // ========================================================================
    
    #[test]
    fn test_boundary_conditions() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Write exactly capacity - 1 (max allowed)
        assert_eq!(rb.write(&vec![1.0; 7]).unwrap(), 7);
        assert_eq!(rb.space(), 0);
        assert_eq!(rb.available(), 7);
        
        // Should fail when full
        assert!(matches!(
            rb.write(&vec![2.0; 1]), 
            Err(RingBufferError::BufferFull)
        ));
        
        // Single element operations at boundary
        let mut single = vec![0.0; 1];
        assert_eq!(rb.read(&mut single).unwrap(), 1);
        assert_eq!(single[0], 1.0);
        assert_eq!(rb.space(), 1);
        
        // Can write one more
        assert_eq!(rb.write(&vec![3.0; 1]).unwrap(), 1);
        assert_eq!(rb.space(), 0);
    }
    
    #[test]
    fn test_single_element_operations() {
        let rb = RingBuffer::new(2).unwrap(); // Minimum useful size
        
        // Single writes and reads
        for i in 0..10 {
            assert_eq!(rb.write(&vec![i as f32]).unwrap(), 1);
            let mut out = vec![0.0];
            assert_eq!(rb.read(&mut out).unwrap(), 1);
            assert_eq!(out[0], i as f32);
        }
    }
    
    // ========================================================================
    // Empty Buffer Tests
    // ========================================================================
    
    #[test]
    fn test_empty_buffer_operations() {
        let rb = RingBuffer::new(8).unwrap();
        let mut output = vec![99.0; 4]; // Non-zero to verify zeroing
        
        // Read from empty should error
        assert!(matches!(
            rb.read(&mut output), 
            Err(RingBufferError::BufferEmpty)
        ));
        
        // Peek from empty returns 0
        assert_eq!(rb.peek(&mut output), 0);
        assert_eq!(output, vec![0.0; 4]); // Should zero-fill
        
        // Skip from empty returns 0
        assert_eq!(rb.skip(10), 0);
        
        // read_available should zero-fill
        let mut output2 = vec![99.0; 4];
        let read = rb.read_available(&mut output2);
        assert_eq!(read, 0);
        assert_eq!(output2, vec![0.0; 4]);
    }
    
    // ========================================================================
    // Partial Operation Tests
    // ========================================================================
    
    #[test]
    fn test_partial_write_operations() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Fill partially
        assert_eq!(rb.write(&vec![1.0; 3]).unwrap(), 3);
        
        // Try to write more than space available
        let written = rb.write_available(&vec![2.0; 10]);
        assert_eq!(written, 4); // Only 4 more fit (7 - 3)
        
        // Verify contents
        let mut all = vec![0.0; 7];
        assert_eq!(rb.read(&mut all).unwrap(), 7);
        assert_eq!(all, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    }
    
    #[test]
    fn test_partial_read_with_zero_fill() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Write less than will be requested
        rb.write(&vec![1.0, 2.0, 3.0]).unwrap();
        
        // Request more than available
        let mut output = vec![99.0; 10]; // Non-zero initial
        let read = rb.read_available(&mut output);
        
        assert_eq!(read, 3);
        assert_eq!(&output[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&output[3..], &[0.0; 7]); // Rest zero-filled
    }
    
    #[test]
    fn test_insufficient_space_errors() {
        let rb = RingBuffer::new(8).unwrap();
        
        // Fill buffer
        rb.write(&vec![1.0; 5]).unwrap();
        
        // Try to write more than available space
        match rb.write(&vec![2.0; 5]) {
            Err(RingBufferError::InsufficientSpace { requested, available }) => {
                assert_eq!(requested, 5);
                assert_eq!(available, 2); // 7 - 5
            }
            _ => panic!("Expected InsufficientSpace error"),
        }
        
        // Try to read more than available
        let mut output = vec![0.0; 10];
        match rb.read(&mut output) {
            Err(RingBufferError::InsufficientSpace { requested, available }) => {
                assert_eq!(requested, 10);
                assert_eq!(available, 5);
            }
            _ => panic!("Expected InsufficientSpace error"),
        }
    }
    
    // ========================================================================
    // Peek and Skip Tests
    // ========================================================================
    
    #[test]
    fn test_peek_operation() {
        let rb = RingBuffer::new(8).unwrap();
        
        rb.write(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        // Peek shouldn't consume
        let mut peeked = vec![0.0; 2];
        assert_eq!(rb.peek(&mut peeked), 2);
        assert_eq!(peeked, vec![1.0, 2.0]);
        assert_eq!(rb.available(), 4); // Still available
        
        // Peek with wraparound
        rb.skip(2); // Now at position 2
        rb.write(&vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let mut peek_wrap = vec![0.0; 6];
        assert_eq!(rb.peek(&mut peek_wrap), 6);
        assert_eq!(peek_wrap, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
    
    #[test]
    fn test_skip_operation() {
        let rb = RingBuffer::new(8).unwrap();
        
        rb.write(&vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        
        // Skip some
        assert_eq!(rb.skip(2), 2);
        assert_eq!(rb.available(), 3);
        
        // Read remaining
        let mut output = vec![0.0; 3];
        rb.read(&mut output).unwrap();
        assert_eq!(output, vec![3.0, 4.0, 5.0]);
        
        // Skip more than available
        rb.write(&vec![6.0, 7.0]).unwrap();
        assert_eq!(rb.skip(10), 2);
        assert!(rb.is_empty());
    }
    
    // ========================================================================
    // CRITICAL: Concurrent SPSC Tests
    // ========================================================================
    
    #[test]
    fn test_concurrent_spsc_basic() {
        let rb = Arc::new(RingBuffer::new(1024).unwrap());
        let rb_consumer = Arc::clone(&rb);
        
        let producer = thread::spawn(move || {
            for i in 0..1000 {
                let data = vec![i as f32];
                while rb.write(&data).is_err() {
                    thread::yield_now();
                }
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut output = vec![0.0; 1];
            for expected in 0..1000 {
                while rb_consumer.read(&mut output).is_err() {
                    thread::yield_now();
                }
                assert_eq!(output[0], expected as f32, "Data corruption at {}", expected);
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
    
    #[test]
    fn test_concurrent_spsc_stress() {
        let rb = Arc::new(RingBuffer::new(256).unwrap());
        let rb_consumer = Arc::clone(&rb);
        
        const ITERATIONS: usize = 100_000;
        const BATCH_SIZE: usize = 17; // Prime number for irregular pattern
        
        let producer = thread::spawn(move || {
            let mut value = 0.0;
            for _ in 0..ITERATIONS {
                let mut data = Vec::with_capacity(BATCH_SIZE);
                for _ in 0..BATCH_SIZE {
                    data.push(value);
                    value += 1.0;
                }
                
                // Keep trying until written
                while rb.write_available(&data) < data.len() {
                    thread::yield_now();
                    
                    // Try to write remaining
                    let written = rb.write_available(&data);
                    if written > 0 {
                        data.drain(..written);
                    }
                }
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut output = vec![0.0; BATCH_SIZE];
            let mut expected = 0.0;
            let mut total_read = 0;
            
            while total_read < ITERATIONS * BATCH_SIZE {
                let read = rb_consumer.read_available(&mut output);
                
                for i in 0..read {
                    assert_eq!(
                        output[i], expected,
                        "Data corruption: expected {}, got {} at position {}",
                        expected, output[i], total_read + i
                    );
                    expected += 1.0;
                }
                
                total_read += read;
                
                if read == 0 {
                    thread::yield_now();
                }
            }
            
            assert_eq!(total_read, ITERATIONS * BATCH_SIZE);
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
    
    #[test]
    fn test_concurrent_wraparound() {
        let rb = Arc::new(RingBuffer::new(64).unwrap()); // Small buffer to force wraparounds
        let rb_consumer = Arc::clone(&rb);
        
        let producer = thread::spawn(move || {
            for batch in 0..1000 {
                let start = batch * 10;
                let data: Vec<f32> = (start..start + 10).map(|i| i as f32).collect();
                
                let mut written = 0;
                while written < data.len() {
                    written += rb.write_available(&data[written..]);
                    if written < data.len() {
                        thread::sleep(Duration::from_micros(1));
                    }
                }
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut expected = 0.0;
            let mut output = vec![0.0; 15]; // Different size than producer
            
            while expected < 10000.0 {
                let read = rb_consumer.read_available(&mut output);
                
                for i in 0..read {
                    assert_eq!(output[i], expected);
                    expected += 1.0;
                }
                
                if read == 0 {
                    thread::sleep(Duration::from_micros(1));
                }
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
    
    // ========================================================================
    // Statistics Tests
    // ========================================================================
    
    #[test]
    fn test_statistics_tracking() {
        let rb = RingBuffer::new(16).unwrap();

        // Track writes
        rb.write(&vec![1.0; 5]).unwrap();
        assert_eq!(rb.stats.total_written.load(Ordering::Relaxed), 5);

        rb.write_available(&vec![2.0; 20]);  // Will only write 10
        assert_eq!(rb.stats.total_written.load(Ordering::Relaxed), 15);

        // Track reads
        let mut output = vec![0.0; 8];
        rb.read(&mut output).unwrap();
        assert_eq!(rb.stats.total_read.load(Ordering::Relaxed), 8);

        rb.skip(3);
        assert_eq!(rb.stats.total_read.load(Ordering::Relaxed), 11);

        // Fill buffer to capacity-1 (15 total)
        rb.read(&mut vec![0.0; 4]).unwrap(); // Read 4 more (leaving 0)
        rb.write(&vec![1.0; 15]).unwrap(); // Write max (15)

        // Now try to write when full - should increment overflow
        rb.write(&vec![3.0; 1]).ok();
        assert!(rb.stats.overflow_count.load(Ordering::Relaxed) > 0);

        // Track underflows
        rb.read(&mut vec![0.0; 15]).unwrap(); // Read all
        rb.read(&mut output).ok(); // Try to read from empty
        assert!(rb.stats.underflow_count.load(Ordering::Relaxed) > 0);
    }

    // ========================================================================
    // StaticRingBuffer Tests
    // ========================================================================
    
    #[test]
    fn test_static_ring_buffer() {
        let rb = StaticRingBuffer::<128>::new();
        
        // Test basic operations
        assert_eq!(rb.capacity(), 128);
        
        let data = vec![1.0; 64];
        assert_eq!(rb.write(&data).unwrap(), 64);
        
        let mut output = vec![0.0; 64];
        assert_eq!(rb.read(&mut output).unwrap(), 64);
        assert_eq!(output, data);
        
        // Test wraparound
        rb.write(&vec![2.0; 100]).unwrap();
        rb.read(&mut output).unwrap();
        rb.write(&vec![3.0; 50]).unwrap();
        
        let mut large_output = vec![0.0; 86];
        rb.read(&mut large_output).unwrap();
        assert_eq!(&large_output[..36], &vec![2.0; 36]);
        assert_eq!(&large_output[36..], &vec![3.0; 50]);
    }
    
    // ========================================================================
    // DoubleBuffer Tests
    // ========================================================================
    
    #[test]
    fn test_double_buffer_atomic_swap() {
        let db = Arc::new(DoubleBuffer::new(4));
        let db2 = Arc::clone(&db);

        // Initial state
        assert_eq!(db.active_index(), 0);
        assert_eq!(db.inactive_index(), 1);

        // Concurrent swapping
        let swapper = thread::spawn(move || {
            for _ in 0..1000 {
                db2.swap();
                thread::yield_now();
            }
        });

        // Verify indices are always valid (but may be same due to race)
        for _ in 0..1000 {
            let active = db.active_index();
            assert!(active == 0 || active == 1);
            thread::yield_now();
        }

        swapper.join().unwrap();

        // We can't assert they're different due to race conditions
    }

    // ========================================================================
    // Memory Ordering Verification (run with Miri)
    // ========================================================================

    #[test]
    #[cfg(miri)]
    fn test_memory_ordering_miri() {
        // This test is specifically for running under Miri
        // cargo +nightly miri test test_memory_ordering_miri
        
        let rb = Arc::new(RingBuffer::new(128).unwrap());
        let rb2 = Arc::clone(&rb);
        
        let producer = thread::spawn(move || {
            for i in 0..100 {
                let data = vec![i as f32; 4];
                rb.write(&data).unwrap();
            }
        });
        
        let consumer = thread::spawn(move || {
            let mut output = vec![0.0; 4];
            for expected in 0..100 {
                while rb2.read(&mut output).is_err() {
                    thread::yield_now();
                }
                assert_eq!(output[0], expected as f32);
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
    
    // ========================================================================
    // Fuzzing Preparation
    // ========================================================================
    
    #[test]
    fn test_random_operations_sequence() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let rb = RingBuffer::new(32).unwrap();
        let mut hasher = DefaultHasher::new();
        
        // Pseudo-random but deterministic sequence
        for i in 0..1000 {
            i.hash(&mut hasher);
            let op = hasher.finish() % 5;
            
            match op {
                0 => {
                    let size = (i % 10) + 1;
                    let data = vec![i as f32; size];
                    rb.write_available(&data);
                }
                1 => {
                    let mut output = vec![0.0; (i % 10) + 1];
                    rb.read_available(&mut output);
                }
                2 => {
                    rb.skip(i % 5);
                }
                3 => {
                    let mut output = vec![0.0; 5];
                    rb.peek(&mut output);
                }
                _ => {
                    // Check invariants
                    assert!(rb.available() <= rb.capacity() - 1);
                    assert!(rb.space() <= rb.capacity() - 1);
                    assert_eq!(rb.available() + rb.space(), rb.capacity() - 1);
                }
            }
        }
    }
}
