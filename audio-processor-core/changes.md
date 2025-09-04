  1. Add Spectral Smoothing:
    - Apply median filter across frequency bins
    - Use Gaussian smoothing with σ = 1-2 bins
  2. Add Temporal Smoothing:
    - IIR filter on gain values: gain[t] = α*gain[t] + (1-α)*gain[t-1]
    - α = 0.7-0.9 for speech
  3. Improve Noise Estimation:
    - Use minimum statistics (track minimum over sliding window)
    - Update continuously, not just from first 0.5s
  4. Add Perceptual Constraints:
    - Limit gain changes to ±6dB per frame
    - Preserve harmonic structure
    - Use psychoacoustic masking
