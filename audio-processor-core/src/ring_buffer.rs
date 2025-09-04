//! Lock-free ring buffer for real-time audio processing
//! 
//! This module provides a high-performance, cache-optimized ring buffer
//! specifically designed for single-producer single-consumer scenarios
//! common in audio processing pipelines.

use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::cell::UnsafeCell;
use std::ptr;
use std::sync::Arc;
use std::marker::PhantomData;

/// Standard cache line size (x86_64/ARM64)
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

/// Internal shared state for the ring buffer
/// This is wrapped in Arc and shared between Producer and Consumer
#[repr(C)]  // Preserve field order to ensure padding stays between atomics
struct Inner {
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

// Safe because we control access through Producer/Consumer
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn new(requested_size: usize) -> Result<Self, RingBufferError> {
        // Reject sizes that are too small to be useful
        if requested_size <= 1 {
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
    
    #[inline]
    const fn capacity(&self) -> usize {
        self.size
    }
}

/// Producer handle for writing to the ring buffer
/// 
/// This handle is Send but not Sync, ensuring exactly one producer thread
pub struct Producer {
    inner: Arc<Inner>,
    // Makes this type !Sync to prevent concurrent access
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// Send allows moving between threads, but !Sync prevents sharing
unsafe impl Send for Producer {}

/// Consumer handle for reading from the ring buffer
/// 
/// This handle is Send but not Sync, ensuring exactly one consumer thread
pub struct Consumer {
    inner: Arc<Inner>,
    // Makes this type !Sync to prevent concurrent access
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// Send allows moving between threads, but !Sync prevents sharing
unsafe impl Send for Consumer {}

/// Create a new SPSC ring buffer with the specified capacity
/// 
/// Returns a Producer and Consumer pair. Each can be moved to different threads
/// but cannot be shared between threads (enforced at compile time).
/// 
/// # Example
/// ```
/// use ring_buffer::spsc_ring_buffer;
/// use std::thread;
/// 
/// let (producer, consumer) = spsc_ring_buffer(1024).unwrap();
/// 
/// // Producer thread
/// let producer_thread = thread::spawn(move || {
///     let data = vec![1.0; 128];
///     producer.write(&data).unwrap();
/// });
/// 
/// // Consumer thread
/// let consumer_thread = thread::spawn(move || {
///     let mut output = vec![0.0; 128];
///     consumer.read(&mut output).unwrap();
/// });
/// ```
pub fn spsc_ring_buffer(capacity: usize) -> Result<(Producer, Consumer), RingBufferError> {
    let inner = Arc::new(Inner::new(capacity)?);
    Ok((
        Producer {
            inner: inner.clone(),
            _not_sync: PhantomData,
        },
        Consumer {
            inner,
            _not_sync: PhantomData,
        },
    ))
}

// ============================================================================
// Producer Implementation
// ============================================================================

impl Producer {
    /// Get the actual size of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    
    /// Get number of samples available for writing
    #[inline]
    pub fn space(&self) -> usize {
        let write = self.inner.write_pos.0.load(Ordering::Relaxed); // Producer owns write_pos
        let read = self.inner.read_pos.0.load(Ordering::Acquire);
        self.inner.size.saturating_sub(write.wrapping_sub(read)).saturating_sub(1)
    }
    
    /// Write a frame of samples
    /// Returns Ok(samples_written) or error
    #[must_use]
    pub fn write(&self, frame: &[f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);
        }

        if frame.len() > self.inner.size {
            return Err(RingBufferError::InvalidSize);
        }

        let write_pos = self.inner.write_pos.0.load(Ordering::Relaxed);
        let read_pos = self.inner.read_pos.0.load(Ordering::Acquire);
        
        let available_space = self.inner.size
            .saturating_sub(write_pos.wrapping_sub(read_pos))
            .saturating_sub(1);
        
        if available_space < frame.len() {
            self.inner.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available: available_space,
            });
        }
        
        // Write data using optimized copy
        unsafe {
            self.write_unchecked(write_pos, frame);
        }
        
        // Update write position with Release to ensure data is visible
        self.inner.write_pos.0.store(write_pos.wrapping_add(frame.len()), Ordering::Release);
        self.inner.stats.total_written.fetch_add(frame.len() as u64, Ordering::Relaxed);
        
        Ok(frame.len())
    }
    
    /// Write samples, partially if necessary
    #[must_use]
    pub fn write_available(&self, frame: &[f32]) -> usize {
        if frame.is_empty() {
            return 0;
        }
        
        let write_pos = self.inner.write_pos.0.load(Ordering::Relaxed);
        let read_pos = self.inner.read_pos.0.load(Ordering::Acquire);
        
        let available_space = self.inner.size
            .saturating_sub(write_pos.wrapping_sub(read_pos))
            .saturating_sub(1);
        let to_write = frame.len().min(available_space);
        
        if to_write == 0 {
            self.inner.stats.overflow_count.fetch_add(1, Ordering::Relaxed);
            return 0;
        }
        
        unsafe {
            self.write_unchecked(write_pos, &frame[..to_write]);
        }
        
        self.inner.write_pos.0.store(write_pos.wrapping_add(to_write), Ordering::Release);
        self.inner.stats.total_written.fetch_add(to_write as u64, Ordering::Relaxed);
        to_write
    }
    
    /// Get a reference to the statistics
    pub fn stats(&self) -> &RingBufferStats {
        &self.inner.stats
    }
    
    /// Internal unchecked write using optimized memory copy
    #[inline(always)]
    unsafe fn write_unchecked(&self, start_pos: usize, data: &[f32]) {
        let len = data.len();
        let start_idx = start_pos & self.inner.mask;
        let first_part_len = (self.inner.size - start_idx).min(len);
        
        // Get raw pointer for direct memory operations
        let buffer_ptr = self.inner.buffer.as_ptr() as *mut f32;
        
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
}

// ============================================================================
// Consumer Implementation
// ============================================================================

impl Consumer {
    /// Get the actual size of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    
    /// Get number of samples available for reading
    #[inline]
    pub fn available(&self) -> usize {
        let write = self.inner.write_pos.0.load(Ordering::Acquire);
        let read = self.inner.read_pos.0.load(Ordering::Relaxed); // Consumer owns read_pos
        write.wrapping_sub(read)
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        let write = self.inner.write_pos.0.load(Ordering::Acquire);
        let read = self.inner.read_pos.0.load(Ordering::Relaxed);
        write == read
    }
    
    /// Read a frame of samples
    /// Returns Ok(samples_read) or error
    #[must_use]
    pub fn read(&self, frame: &mut [f32]) -> Result<usize, RingBufferError> {
        if frame.is_empty() {
            return Ok(0);
        }
                
        let read_pos = self.inner.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.inner.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        
        if available < frame.len() {
            self.inner.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::InsufficientSpace {
                requested: frame.len(),
                available,
            });
        }
        
        // Read data using optimized copy
        unsafe {
            self.read_unchecked(read_pos, frame);
        }
        
        // Update read position
        self.inner.read_pos.0.fetch_add(frame.len(), Ordering::Release);
        self.inner.stats.total_read.fetch_add(frame.len() as u64, Ordering::Relaxed);
        
        Ok(frame.len())
    }
    
    /// Read available samples, fill rest with zeros
    #[must_use]
    pub fn read_available(&self, frame: &mut [f32]) -> usize {
        if frame.is_empty() {
            return 0;
        }
       
        let read_pos = self.inner.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.inner.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        let to_read = frame.len().min(available);
        
        if to_read == 0 {
            frame.fill(0.0);
            self.inner.stats.underflow_count.fetch_add(1, Ordering::Relaxed);
            return 0;
        }
        
        unsafe {
            self.read_unchecked(read_pos, &mut frame[..to_read]);
        }
        
        // Fill rest with silence
        if to_read < frame.len() {
            frame[to_read..].fill(0.0);
        }
        
        self.inner.read_pos.0.fetch_add(to_read, Ordering::Release);
        self.inner.stats.total_read.fetch_add(to_read as u64, Ordering::Relaxed);
        to_read
    }
    
    /// Peek at samples without consuming them
    #[must_use]
    pub fn peek(&self, output: &mut [f32]) -> usize {
        let read_pos = self.inner.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.inner.write_pos.0.load(Ordering::Acquire);
        
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
    
    /// Skip samples without reading them
    #[must_use]
    pub fn skip(&self, count: usize) -> usize {
        let read_pos = self.inner.read_pos.0.load(Ordering::Relaxed);
        let write_pos = self.inner.write_pos.0.load(Ordering::Acquire);
        
        let available = write_pos.wrapping_sub(read_pos);
        let to_skip = count.min(available);
        
        if to_skip > 0 {
            self.inner.read_pos.0.fetch_add(to_skip, Ordering::Release);
            self.inner.stats.total_read.fetch_add(to_skip as u64, Ordering::Relaxed);
        }
        
        to_skip
    }
    
    /// Get a reference to the statistics
    pub fn stats(&self) -> &RingBufferStats {
        &self.inner.stats
    }
    
    /// Internal unchecked read using optimized memory copy
    #[inline(always)]
    unsafe fn read_unchecked(&self, start_pos: usize, data: &mut [f32]) {
        let len = data.len();
        let start_idx = start_pos & self.inner.mask;
        let first_part_len = (self.inner.size - start_idx).min(len);
        
        // Get raw pointer for direct memory operations
        let buffer_ptr = self.inner.buffer.as_ptr() as *const f32;
        
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
}

/// Static-sized ring buffer for compile-time known sizes
/// Provides better optimization opportunities
pub struct StaticRingBuffer<const SIZE: usize> {
    producer: Producer,
    consumer: Consumer,
}

impl<const SIZE: usize> StaticRingBuffer<SIZE> {
    /// Create a new static ring buffer
    /// Size must be a power of 2 (checked at runtime for now)
    pub fn new() -> Result<(StaticProducer<SIZE>, StaticConsumer<SIZE>), RingBufferError> {
        // Runtime assertions (compile-time const assertions not supported in generic contexts)
        if !SIZE.is_power_of_two() {
            return Err(RingBufferError::InvalidSize);
        }
        if SIZE <= 1 {
            return Err(RingBufferError::InvalidSize);
        }
        
        let (producer, consumer) = spsc_ring_buffer(SIZE)?;
        Ok((
            StaticProducer { inner: producer },
            StaticConsumer { inner: consumer },
        ))
    }
}

/// Static producer with compile-time size
pub struct StaticProducer<const SIZE: usize> {
    inner: Producer,
}

unsafe impl<const SIZE: usize> Send for StaticProducer<SIZE> {}

impl<const SIZE: usize> StaticProducer<SIZE> {
    #[inline]
    pub const fn capacity(&self) -> usize {
        SIZE
    }
    
    #[inline]
    pub fn space(&self) -> usize {
        self.inner.space()
    }
    
    #[inline]
    pub fn write(&self, frame: &[f32]) -> Result<usize, RingBufferError> {
        self.inner.write(frame)
    }
    
    #[inline]
    pub fn write_available(&self, frame: &[f32]) -> usize {
        self.inner.write_available(frame)
    }
}

/// Static consumer with compile-time size
pub struct StaticConsumer<const SIZE: usize> {
    inner: Consumer,
}

unsafe impl<const SIZE: usize> Send for StaticConsumer<SIZE> {}

impl<const SIZE: usize> StaticConsumer<SIZE> {
    #[inline]
    pub const fn capacity(&self) -> usize {
        SIZE
    }
    
    #[inline]
    pub fn available(&self) -> usize {
        self.inner.available()
    }
    
    #[inline]
    pub fn read(&self, frame: &mut [f32]) -> Result<usize, RingBufferError> {
        self.inner.read(frame)
    }
    
    #[inline]
    pub fn read_available(&self, frame: &mut [f32]) -> usize {
        self.inner.read_available(frame)
    }
    
    #[inline]
    pub fn peek(&self, output: &mut [f32]) -> usize {
        self.inner.peek(output)
    }
    
    #[inline]
    pub fn skip(&self, count: usize) -> usize {
        self.inner.skip(count)
    }
}

/// Double buffer for smooth audio processing with atomic swapping
/// 
/// # Safety
/// The `active_mut` method requires careful synchronization.
/// No swaps should occur while holding a mutable reference.
#[repr(C)]  // Preserve field order
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
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_basic_operations() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        // Initial state
        assert_eq!(consumer.available(), 0);
        assert_eq!(producer.space(), 7);
        assert!(consumer.is_empty());
        
        // Write and verify
        let input = vec![1.0, 2.0, 3.0];
        assert_eq!(producer.write(&input).unwrap(), 3);
        assert_eq!(consumer.available(), 3);
        assert_eq!(producer.space(), 4);
        assert!(!consumer.is_empty());
        
        // Read and verify
        let mut output = vec![0.0; 3];
        assert_eq!(consumer.read(&mut output).unwrap(), 3);
        assert_eq!(output, input);
        assert!(consumer.is_empty());
    }
    
    #[test]
    fn test_size_validation() {
        // Size 0 and 1 should be rejected
        assert!(matches!(spsc_ring_buffer(0), Err(RingBufferError::InvalidSize)));
        assert!(matches!(spsc_ring_buffer(1), Err(RingBufferError::InvalidSize)));
        
        // Size 2 and above should work
        assert!(spsc_ring_buffer(2).is_ok());
        assert!(spsc_ring_buffer(8).is_ok());
    }
    
    #[test]
    fn test_wraparound() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        // Fill most of buffer
        assert_eq!(producer.write(&vec![1.0; 6]).unwrap(), 6);
        
        // Read some to make space at beginning
        let mut temp = vec![0.0; 4];
        assert_eq!(consumer.read(&mut temp).unwrap(), 4);
        assert_eq!(temp, vec![1.0; 4]);
        
        // Write across boundary
        assert_eq!(producer.write(&vec![2.0; 5]).unwrap(), 5);
        
        // Read across boundary
        let mut result = vec![0.0; 7];
        assert_eq!(consumer.read(&mut result).unwrap(), 7);
        assert_eq!(result, vec![1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }
    
    #[test]
    fn test_concurrent_spsc() {
        let (producer, consumer) = spsc_ring_buffer(1024).unwrap();
        
        let producer_thread = thread::spawn(move || {
            for i in 0..1000 {
                let data = vec![i as f32];
                while producer.write(&data).is_err() {
                    thread::yield_now();
                }
            }
        });
        
        let consumer_thread = thread::spawn(move || {
            let mut output = vec![0.0; 1];
            for expected in 0..1000 {
                while consumer.read(&mut output).is_err() {
                    thread::yield_now();
                }
                assert_eq!(output[0], expected as f32);
            }
        });
        
        producer_thread.join().unwrap();
        consumer_thread.join().unwrap();
    }
    
    #[test]
    fn test_producer_faster_than_consumer_sample_loss() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        // Fill the buffer completely (7 samples max)
        assert_eq!(producer.write(&vec![1.0; 7]).unwrap(), 7);
        assert_eq!(producer.space(), 0);
        
        // Try to write when full - should fail and indicate overflow
        let result = producer.write(&vec![2.0; 1]);
        assert!(matches!(result, Err(RingBufferError::InsufficientSpace { .. })));
        
        // Verify stats tracked the overflow
        let stats = producer.stats();
        assert_eq!(stats.overflow_count.load(Ordering::Relaxed), 1);
        
        // Consumer reads some data
        let mut temp = vec![0.0; 3];
        assert_eq!(consumer.read(&mut temp).unwrap(), 3);
        
        // Now producer can write again
        assert_eq!(producer.write(&vec![3.0; 3]).unwrap(), 3);
        
        // Verify write tracking
        assert_eq!(stats.total_written.load(Ordering::Relaxed), 10); // 7 + 3
    }
    
    #[test]
    fn test_space_available_consistency() {
        let (producer, consumer) = spsc_ring_buffer(16).unwrap();
        
        // Initial space should be size - 1
        assert_eq!(producer.space(), 15);
        
        // Write some data
        assert_eq!(producer.write(&vec![1.0; 5]).unwrap(), 5);
        assert_eq!(producer.space(), 10);
        
        // Test across wraparound
        assert_eq!(consumer.read(&mut vec![0.0; 3]).unwrap(), 3);
        assert_eq!(producer.space(), 13);
        
        // Fill to near capacity
        assert_eq!(producer.write(&vec![2.0; 13]).unwrap(), 13);
        assert_eq!(producer.space(), 0);
        
        // Space calculation should be consistent at boundaries
        assert_eq!(consumer.read(&mut vec![0.0; 1]).unwrap(), 1);
        assert_eq!(producer.space(), 1);
    }
    
    #[test]
    fn test_wraparound_boundaries() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        // Test writing exactly at wraparound point
        assert_eq!(producer.write(&vec![1.0; 7]).unwrap(), 7);
        assert_eq!(consumer.read(&mut vec![0.0; 7]).unwrap(), 7);
        
        // Write should wrap to beginning
        assert_eq!(producer.write(&vec![2.0; 4]).unwrap(), 4);
        assert_eq!(consumer.read(&mut vec![0.0; 4]).unwrap(), 4);
        
        // Test partial writes across boundary
        assert_eq!(producer.write(&vec![3.0; 6]).unwrap(), 6);
        let mut output = vec![0.0; 6];
        assert_eq!(consumer.read(&mut output).unwrap(), 6);
        assert_eq!(output, vec![3.0; 6]);
    }
    
    #[test]
    fn test_multi_thread_spsc_stress() {
        let (producer, consumer) = spsc_ring_buffer(256).unwrap();
        
        // Producer thread - aggressive writing
        let producer_thread = thread::spawn(move || {
            let mut total_written = 0u64;
            let mut overflow_count = 0u64;
            let data = vec![1.0; 17]; // Prime number size for irregular patterns
            
            for _ in 0..10000 {
                match producer.write(&data) {
                    Ok(n) => total_written += n as u64,
                    Err(_) => {
                        overflow_count += 1;
                        // Buffer full, spin briefly
                        thread::yield_now();
                    }
                }
            }
            (total_written, overflow_count)
        });
        
        // Consumer thread - slower reading
        let consumer_thread = thread::spawn(move || {
            let mut total_read = 0u64;
            let mut buffer = vec![0.0; 13]; // Different size for testing
            
            // Read with small delays to simulate slower consumer
            for _ in 0..50000 {
                match consumer.read(&mut buffer) {
                    Ok(n) => {
                        total_read += n as u64;
                        // Simulate processing time
                        std::hint::black_box(&buffer);
                    }
                    Err(_) => {
                        // Buffer empty
                        thread::yield_now();
                    }
                }
            }
            total_read
        });
        
        let (written, overflows) = producer_thread.join().unwrap();
        let read = consumer_thread.join().unwrap();
        
        println!("Stress test: written={}, read={}, overflows={}", 
                 written, read, overflows);
        
        // Some overflows are expected with aggressive producer
        assert!(overflows > 0, "Expected some overflows with aggressive producer");
        
        // Written should be positive
        assert!(written > 0, "Should have written some data");
    }
    
    #[test]
    fn test_partial_operations() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        // Fill partially
        assert_eq!(producer.write(&vec![1.0; 3]).unwrap(), 3);
        
        // Try to write more than space available
        let written = producer.write_available(&vec![2.0; 10]);
        assert_eq!(written, 4); // Only 4 more fit
        
        // Read with zero-fill
        let mut output = vec![99.0; 10];
        let read = consumer.read_available(&mut output);
        assert_eq!(read, 7);
        assert_eq!(&output[0..7], &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(&output[7..], &[0.0; 3]);
    }
    
    #[test]
    fn test_peek_and_skip() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        producer.write(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        // Peek shouldn't consume
        let mut peeked = vec![0.0; 2];
        assert_eq!(consumer.peek(&mut peeked), 2);
        assert_eq!(peeked, vec![1.0, 2.0]);
        assert_eq!(consumer.available(), 4);
        
        // Skip some
        assert_eq!(consumer.skip(2), 2);
        assert_eq!(consumer.available(), 2);
        
        // Read remaining
        let mut output = vec![0.0; 2];
        consumer.read(&mut output).unwrap();
        assert_eq!(output, vec![3.0, 4.0]);
    }
    
    #[test]
    fn test_stats() {
        let (producer, consumer) = spsc_ring_buffer(8).unwrap();
        
        producer.write(&vec![1.0; 4]).unwrap();
        assert_eq!(producer.stats().total_written.load(Ordering::Relaxed), 4);
        
        let mut output = vec![0.0; 4];
        consumer.read(&mut output).unwrap();
        assert_eq!(consumer.stats().total_read.load(Ordering::Relaxed), 4);
        
        // Fill to trigger overflow tracking
        producer.write(&vec![1.0; 7]).unwrap();
        producer.write(&vec![1.0; 1]).ok();
        assert!(producer.stats().overflow_count.load(Ordering::Relaxed) > 0);
    }
    
    // The following would fail to compile, demonstrating type safety:
    // #[test]
    // fn compile_fail_shared_producer() {
    //     let (producer, _consumer) = spsc_ring_buffer(8).unwrap();
    //     let p1 = &producer;
    //     let p2 = &producer;
    //     
    //     thread::spawn(move || {
    //         p1.write(&vec![1.0]);  // Error: Producer is !Sync
    //     });
    //     thread::spawn(move || {
    //         p2.write(&vec![2.0]);  // Can't share across threads
    //     });
    // }
}
