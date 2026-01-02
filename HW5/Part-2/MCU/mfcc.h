/* mfcc.h (changes) */
#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>
#include "arm_math.h"

typedef struct {
    // Configuration
    uint32_t fft_len;
    uint32_t num_mel_filters;
    uint32_t num_dct_coeffs;

    // Pre-computed tables
    float* dct_matrix;                 // size: num_dct_coeffs * num_mel_filters
    float* window_coeffs;              // size: fft_len
    float* packed_mel_filters;         // packed triangle weights
    uint32_t* mel_filter_start_indices;// start index into spectrum (clipped)
    uint32_t* mel_filter_lengths;      // length for each mel filter

    // CMSIS-DSP FFT instance
    arm_rfft_fast_instance_f32* rfft_instance;

    // Large scratch buffers allocated on heap (to avoid stack overflow)
    float* scratch_time;               // size: fft_len (windowed input)
    float* scratch_fft_out;            // size: fft_len (rfft output raw)
    float* scratch_power;              // size: fft_len/2 + 1
    float* scratch_mel_energies;       // size: num_mel_filters
} MfccInstance;

MfccInstance* mfcc_create_instance(uint32_t fft_len, uint32_t num_mel_filters, uint32_t num_dct_coeffs);
void mfcc_destroy_instance(MfccInstance* instance);
void mfcc_compute(const MfccInstance* instance, const int16_t* audio_frame, float* mfcc_output);

#endif // MFCC_H
