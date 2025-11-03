#include "mfcc.h"
#include <cmath> // For standard math functions like logf, cosf, sqrtf

// --- Mel Scale Conversion Formulas ---
float Mfcc::frequency_to_mel(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
}

float Mfcc::mel_to_frequency(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

// --- Constructor: Initializes the MFCC processor ---
Mfcc::Mfcc(uint32_t fft_len, uint32_t num_mel_filters, uint32_t num_dct_outputs)
    : fft_len_(fft_len),
      num_mel_filters_(num_mel_filters),
      num_dct_outputs_(num_dct_outputs) {
    
    // 1. Pre-compute the Hamming window coefficients
    create_window();
    // 2. Pre-compute the Mel filter bank
    create_mel_fbank();
    // 3. Pre-compute the DCT matrix
    create_dct_matrix();
    // 4. Initialize the CMSIS-DSP FFT instance
    arm_rfft_fast_init_f32(&rfft_instance_, fft_len_);
}

// --- Initialization Helper Methods ---

void Mfcc::create_window() {
    window_coeffs_.resize(fft_len_);
    for (uint32_t i = 0; i < fft_len_; i++) {
        window_coeffs_[i] = 0.5f - 0.5f * cosf(2.0f * PI * static_cast<float>(i) / (fft_len_));
    }
}

void Mfcc::create_dct_matrix() {
    dct_matrix_.resize(num_dct_outputs_ * num_mel_filters_);
    float norm_mels = sqrtf(2.0f / num_mel_filters_);
    for (uint32_t dct_idx = 0; dct_idx < num_dct_outputs_; dct_idx++) {
        for (uint32_t mel_idx = 0; mel_idx < num_mel_filters_; mel_idx++) {
            float s = (static_cast<float>(mel_idx) + 0.5f) / num_mel_filters_;
            dct_matrix_[dct_idx * num_mel_filters_ + mel_idx] = cosf(static_cast<float>(dct_idx) * PI * s) * norm_mels;
        }
    }
}

void Mfcc::create_mel_fbank() {
    const uint32_t half_fft_size = fft_len_ / 2;
    const float fmin_mel = frequency_to_mel(MEL_LOW_FREQ);
    const float fmax_mel = frequency_to_mel(MEL_HIGH_FREQ);
    const float mel_step = (fmax_mel - fmin_mel) / (num_mel_filters_ + 1);

    filter_pos_.resize(num_mel_filters_);
    filter_lengths_.resize(num_mel_filters_);

    for (uint32_t mel_idx = 0; mel_idx < num_mel_filters_; mel_idx++) {
        float mel_center = fmin_mel + (mel_idx + 1) * mel_step;
        float mel_left = mel_center - mel_step;
        float mel_right = mel_center + mel_step;

        std::vector<float> current_filter;
        bool start_found = false;

        for (uint32_t freq_idx = 0; freq_idx < half_fft_size; ++freq_idx) {
            float linear_freq = static_cast<float>(freq_idx * SAMPLE_RATE) / fft_len_;
            float mel_freq = frequency_to_mel(linear_freq);
            
            float filter_val = 0.0f;
            if (mel_freq > mel_left && mel_freq < mel_right) {
                if (mel_freq < mel_center) {
                    filter_val = (mel_freq - mel_left) / (mel_center - mel_left);
                } else {
                    filter_val = (mel_right - mel_freq) / (mel_right - mel_center);
                }
            }

            if (filter_val > 0.0f) {
                if (!start_found) {
                    filter_pos_[mel_idx] = freq_idx;
                    start_found = true;
                }
                current_filter.push_back(filter_val);
            } else if (start_found) {
                break; // We've passed the non-zero part of the filter
            }
        }
        filter_lengths_[mel_idx] = current_filter.size();
        packed_mel_filters_.insert(packed_mel_filters_.end(), current_filter.begin(), current_filter.end());
    }
}

// --- Main Computation Logic ---

std::vector<float> Mfcc::compute(const std::vector<int16_t>& audio_frame) {
    std::vector<float> frame(fft_len_);
    std::vector<float> buffer(fft_len_);
    std::vector<float> mel_energies(num_mel_filters_);

    // 1. Normalize audio from int16 to float and apply window
    for (uint32_t i = 0; i < fft_len_; i++) {
        frame[i] = static_cast<float>(audio_frame[i]) / 32768.0f;
        frame[i] *= window_coeffs_[i];
    }

    // 2. Compute FFT using CMSIS-DSP (operates on raw pointers from .data())
    arm_rfft_fast_f32(&rfft_instance_, frame.data(), buffer.data(), 0);

    // 3. Compute Power Spectrum (Magnitude Squared)
    const uint32_t half_dim = fft_len_ / 2;
    float first_energy = buffer[0] * buffer[0];
    float last_energy = buffer[1] * buffer[1];
    for (uint32_t i = 1; i < half_dim; i++) {
        float real = buffer[i * 2];
        float imag = buffer[i * 2 + 1];
        buffer[i] = real * real + imag * imag;
    }
    buffer[0] = first_energy;
    buffer[half_dim] = last_energy;

    // 4. Apply Mel Filterbank to the power spectrum
    uint32_t packed_filter_offset = 0;
    for (uint32_t bin = 0; bin < num_mel_filters_; bin++) {
        float mel_energy = 0.0f;
        uint32_t first_index = filter_pos_[bin];
        uint32_t length = filter_lengths_[bin];
        for (uint32_t i = 0; i < length; i++) {
            mel_energy += buffer[first_index + i] * packed_mel_filters_[packed_filter_offset + i];
        }
        mel_energies[bin] = mel_energy;
        packed_filter_offset += length;
    }

    // 5. Take the logarithm of the Mel energies
    for (uint32_t bin = 0; bin < num_mel_filters_; bin++) {
        if (mel_energies[bin] <= 0.0f) {
            mel_energies[bin] = 1e-7f; // Avoid log(0) or log(negative)
        }
        mel_energies[bin] = logf(mel_energies[bin]);
    }
    
    // 6. Compute DCT to get the final MFCC coefficients
    std::vector<float> mfcc_out(num_dct_outputs_);
    for (uint32_t i = 0; i < num_dct_outputs_; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < num_mel_filters_; j++) {
            sum += dct_matrix_[i * num_mel_filters_ + j] * mel_energies[j];
        }
        mfcc_out[i] = sum;
    }

    return mfcc_out;
}