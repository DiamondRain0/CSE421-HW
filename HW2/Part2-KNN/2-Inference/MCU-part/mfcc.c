/* mfcc.c (core functions) */
#include "mfcc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define M_PI 3.14157
/* Local helpers */
static float frequency_to_mel(float freq) {
    return 2595.0f * log10f(1.0f + freq / 700.0f);
}
static float mel_to_frequency(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static float* create_dct_matrix(int num_dct_coeffs, int num_mel_filters) {
    float* matrix = (float*)calloc((size_t)num_dct_coeffs * num_mel_filters, sizeof(float));
    if (!matrix) return NULL;

    float normalizer = sqrtf(2.0f / (float)num_mel_filters);
    for (int k = 0; k < num_dct_coeffs; k++) {
        for (int n = 0; n < num_mel_filters; n++) {
            matrix[k * num_mel_filters + n] = normalizer * cosf((float)M_PI / (float)num_mel_filters * (n + 0.5f) * k);
        }
    }
    return matrix;
}

/* Create mel filter bank: writes to instance->packed_mel_filters, mel_filter_start_indices, mel_filter_lengths */
static int create_mel_fbank(MfccInstance* instance, uint32_t samp_freq, uint32_t mel_low_freq, uint32_t mel_high_freq) {
    uint32_t half_fft_len = instance->fft_len / 2;
    float f_min_mel = frequency_to_mel((float)mel_low_freq);
    float f_max_mel = frequency_to_mel((float)mel_high_freq);
    float mel_step = (f_max_mel - f_min_mel) / (instance->num_mel_filters + 1);
    float freq_step = (float)samp_freq / (float)instance->fft_len;

    uint32_t points = instance->num_mel_filters + 2;
    float* mel_points = (float*)malloc(points * sizeof(float));
    uint32_t* fft_bin_points = (uint32_t*)malloc(points * sizeof(uint32_t));
    if (!mel_points || !fft_bin_points) {
        free(mel_points); free(fft_bin_points);
        return -1;
    }

    for (uint32_t i = 0; i < points; i++) {
        mel_points[i] = f_min_mel + (float)i * mel_step;
        float freq = mel_to_frequency(mel_points[i]);
        /* map to fft bin and clip to half spectrum */
        uint32_t bin = (uint32_t)(freq / freq_step + 0.5f);
        if (bin > half_fft_len) bin = half_fft_len;
        fft_bin_points[i] = bin;
    }

    /* compute lengths and total size */
    uint32_t total_filter_len = 0;
    for (uint32_t i = 0; i < instance->num_mel_filters; i++) {
        uint32_t start = fft_bin_points[i];
        uint32_t center = fft_bin_points[i+1];
        uint32_t end = fft_bin_points[i+2];

        if (end <= start) {
            instance->mel_filter_lengths[i] = 0;
            instance->mel_filter_start_indices[i] = 0;
            continue;
        }

        /* Clip end to half_fft_len */
        if (end > half_fft_len) end = half_fft_len;
        uint32_t len = end - start;
        instance->mel_filter_lengths[i] = len;
        instance->mel_filter_start_indices[i] = start;
        total_filter_len += len;
    }

    instance->packed_mel_filters = (float*)calloc((size_t)total_filter_len, sizeof(float));
    if (!instance->packed_mel_filters) {
        free(mel_points); free(fft_bin_points);
        return -1;
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < instance->num_mel_filters; i++) {
        uint32_t start = fft_bin_points[i];
        uint32_t center = fft_bin_points[i+1];
        uint32_t end = fft_bin_points[i+2];

        /* Clip to half_fft_len */
        if (center > half_fft_len) center = half_fft_len;
        if (end > half_fft_len) end = half_fft_len;

        uint32_t len = instance->mel_filter_lengths[i];
        for (uint32_t j = 0; j < len; j++) {
            uint32_t bin = start + j;
            float weight = 0.0f;
            if (bin <= center && center > start) {
                weight = (float)(bin - start) / (float)(center - start);
            } else if (bin > center && end > center) {
                weight = (float)(end - bin) / (float)(end - center);
            } else {
                weight = 0.0f;
            }
            instance->packed_mel_filters[offset + j] = weight;
        }
        offset += len;
    }

    free(mel_points);
    free(fft_bin_points);
    return 0;
}

MfccInstance* mfcc_create_instance(uint32_t fft_len, uint32_t num_mel_filters, uint32_t num_dct_coeffs) {
    printf("[MFCC_DEBUG] Starting mfcc_create_instance...\n");

    MfccInstance* instance = (MfccInstance*)calloc(1, sizeof(MfccInstance));
    if (!instance) {
        printf("  [MFCC_DEBUG] calloc(MfccInstance) FAILED\n");
        return NULL;
    }

    instance->fft_len = fft_len;
    instance->num_mel_filters = num_mel_filters;
    instance->num_dct_coeffs = num_dct_coeffs;

    printf("  [MFCC_DEBUG] Allocating window_coeffs (%u bytes)...\n", (unsigned)(fft_len * sizeof(float)));
    instance->window_coeffs = (float*)calloc((size_t)fft_len, sizeof(float));
    if (!instance->window_coeffs) { printf("  [MFCC_DEBUG] window_coeffs FAILED\n"); mfcc_destroy_instance(instance); return NULL; }

    printf("  [MFCC_DEBUG] Allocating mel index arrays (%u bytes each)...\n", (unsigned)(num_mel_filters * sizeof(uint32_t)));
    instance->mel_filter_start_indices = (uint32_t*)calloc((size_t)num_mel_filters, sizeof(uint32_t));
    instance->mel_filter_lengths = (uint32_t*)calloc((size_t)num_mel_filters, sizeof(uint32_t));
    if (!instance->mel_filter_start_indices || !instance->mel_filter_lengths) { printf("  [MFCC_DEBUG] mel index alloc FAILED\n"); mfcc_destroy_instance(instance); return NULL; }

    printf("  [MFCC_DEBUG] Allocating rfft_instance struct (%u bytes)...\n", (unsigned)sizeof(arm_rfft_fast_instance_f32));
    instance->rfft_instance = (arm_rfft_fast_instance_f32*)malloc(sizeof(arm_rfft_fast_instance_f32));
    if (!instance->rfft_instance) { printf("  [MFCC_DEBUG] rfft_instance FAILED\n"); mfcc_destroy_instance(instance); return NULL; }

    /* allocate scratch buffers on heap to avoid stack overflow */
    printf("  [MFCC_DEBUG] Allocating scratch buffers...\n");
    instance->scratch_time = (float*)calloc((size_t)fft_len, sizeof(float));
    instance->scratch_fft_out = (float*)calloc((size_t)fft_len, sizeof(float));
    instance->scratch_power = (float*)calloc((size_t)(fft_len / 2 + 1), sizeof(float));
    instance->scratch_mel_energies = (float*)calloc((size_t)num_mel_filters, sizeof(float));
    if (!instance->scratch_time || !instance->scratch_fft_out || !instance->scratch_power || !instance->scratch_mel_energies) {
        printf("  [MFCC_DEBUG] scratch buffer allocation FAILED\n");
        mfcc_destroy_instance(instance);
        return NULL;
    }

    /* window */
    for (uint32_t i = 0; i < fft_len; i++) {
        instance->window_coeffs[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)i / (float)(fft_len - 1));
    }

    /* DCT */
    instance->dct_matrix = create_dct_matrix((int)num_dct_coeffs, (int)num_mel_filters);
    if (!instance->dct_matrix) { printf("  [MFCC_DEBUG] dct_matrix FAILED\n"); mfcc_destroy_instance(instance); return NULL; }

    /* Mel fbank */
    if (create_mel_fbank(instance, /*samp_freq*/8000u, /*mel_low*/20u, /*mel_high*/4000u) != 0) {
        printf("  [MFCC_DEBUG] create_mel_fbank FAILED\n");
        mfcc_destroy_instance(instance);
        return NULL;
    }

    /* init FFT */
    if (arm_rfft_fast_init_f32(instance->rfft_instance, fft_len) != ARM_MATH_SUCCESS) {
        printf("  [MFCC_DEBUG] arm_rfft_fast_init_f32 FAILED\n");
        mfcc_destroy_instance(instance);
        return NULL;
    }

    printf("[MFCC_DEBUG] mfcc_create_instance finished successfully.\n");
    return instance;
}

void mfcc_destroy_instance(MfccInstance* instance) {
    if (!instance) return;
    free(instance->dct_matrix);
    free(instance->window_coeffs);
    free(instance->packed_mel_filters);
    free(instance->mel_filter_start_indices);
    free(instance->mel_filter_lengths);
    free(instance->rfft_instance);
    free(instance->scratch_time);
    free(instance->scratch_fft_out);
    free(instance->scratch_power);
    free(instance->scratch_mel_energies);
    free(instance);
}

/* mfcc_compute uses instance scratch buffers instead of stack arrays */
void mfcc_compute(const MfccInstance* instance, const int16_t* audio_frame, float* mfcc_output) {
    uint32_t fft_len = instance->fft_len;
    uint32_t half_fft_len = fft_len / 2;

    /* window + convert to float (use scratch_time) */
    for (uint32_t i = 0; i < fft_len; i++) {
        instance->scratch_time[i] = (float)audio_frame[i] * instance->window_coeffs[i];
    }

    /* real FFT: scratch_time -> scratch_fft_out */
    arm_rfft_fast_f32(instance->rfft_instance, instance->scratch_time, instance->scratch_fft_out, 0);

    /* compute power spectrum into scratch_power */
    instance->scratch_power[0] = instance->scratch_fft_out[0] * instance->scratch_fft_out[0];
    for (uint32_t i = 1; i < half_fft_len; i++) {
        float real = instance->scratch_fft_out[2*i];
        float imag = instance->scratch_fft_out[2*i + 1];
        instance->scratch_power[i] = real*real + imag*imag;
    }
    instance->scratch_power[half_fft_len] = instance->scratch_fft_out[1] * instance->scratch_fft_out[1];

    /* compute mel energies */
    uint32_t filter_offset = 0;
    for (uint32_t m = 0; m < instance->num_mel_filters; m++) {
        float acc = 0.0f;
        uint32_t len = instance->mel_filter_lengths[m];
        uint32_t start = instance->mel_filter_start_indices[m];
        for (uint32_t j = 0; j < len; j++) {
            uint32_t idx = start + j;
            if (idx > half_fft_len) break;
            acc += instance->scratch_power[idx] * instance->packed_mel_filters[filter_offset + j];
        }
        instance->scratch_mel_energies[m] = acc < 1e-12f ? 1e-12f : acc;
        instance->scratch_mel_energies[m] = logf(instance->scratch_mel_energies[m]);
        filter_offset += len;
    }

    /* DCT: scratch_mel_energies -> mfcc_output using dct_matrix */
    for (uint32_t k = 0; k < instance->num_dct_coeffs; k++) {
        float sum = 0.0f;
        for (uint32_t n = 0; n < instance->num_mel_filters; n++) {
            sum += instance->dct_matrix[k * instance->num_mel_filters + n] * instance->scratch_mel_energies[n];
        }
        mfcc_output[k] = sum;
    }
}
