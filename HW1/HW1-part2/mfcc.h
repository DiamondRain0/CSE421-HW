#ifndef MFCC_H
#define MFCC_H

#include <vector>
#include <cstdint>
#include <cmath>          // for cosf()
#include "arm_math.h"     // CMSIS-DSP types and functions

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// This class requires the ARM CMSIS-DSP library.
class Mfcc {
public:
    /**
     * @brief Constructor for the MFCC feature extractor.
     *        All necessary data (windows, filters, etc.) is pre-computed here.
     * @param fft_len The length of the FFT, which is also the audio frame size.
     * @param num_mel_filters The number of Mel filters to apply to the spectrum.
     * @param num_dct_outputs The final number of MFCC coefficients to produce.
     */
    Mfcc(uint32_t fft_len, uint32_t num_mel_filters, uint32_t num_dct_outputs);

    /**
     * @brief Computes MFCC features for a given frame of audio data.
     * @param audio_frame A vector of int16_t audio samples. Its size must be equal to fft_len.
     * @return A vector of floats containing the computed MFCC coefficients.
     */
    std::vector<float> compute(const std::vector<int16_t>& audio_frame);

    // Default destructor is sufficient as std::vector handles its own memory.
    ~Mfcc() = default;

private:
    // --- Configuration (set in constructor) ---
    const uint32_t fft_len_;
    const uint32_t num_mel_filters_;
    const uint32_t num_dct_outputs_;

    // --- Pre-computed data structures ---
    std::vector<float> window_coeffs_;
    std::vector<float> dct_matrix_;
    std::vector<uint32_t> filter_pos_;
    std::vector<uint32_t> filter_lengths_;
    std::vector<float> packed_mel_filters_;

    // --- CMSIS-DSP FFT instance ---
    arm_rfft_fast_instance_f32 rfft_instance_;

    // --- Private helper methods for initialization ---
    void create_window();
    void create_dct_matrix();
    void create_mel_fbank();

    // --- Static helper functions for Mel scale conversion ---
    static float frequency_to_mel(float freq);
    static float mel_to_frequency(float mel);

public:
    // --- Public Constants ---
    static constexpr int SAMPLE_RATE = 8000;
    static constexpr int MEL_LOW_FREQ = 20;
    static constexpr int MEL_HIGH_FREQ = 4000; // Nyquist frequency for 8kHz
};

#endif // MFCC_H
