
#include "main.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Include feature extraction and model headers
#include "mfcc.h"
#include "mfcc_neuron_config.h" // Defines NUM_FEATURES

// --- Global Configuration ---
#define SAMPLE_RATE 8000U
#define FFT_LEN 1024U
#define HOP_LEN 512U
#define NUM_MEL_FILTERS 20U

// Input Buffer Size (8000 samples * 2 bytes/sample = 16000 bytes)
#define AUDIO_SAMPLES SAMPLE_RATE 

#define READY_SIGNAL "[READY]\n"

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

#ifdef __GNUC__
int _write(int file, char *ptr, int len) { HAL_UART_Transmit(APP_UART, (uint8_t*)ptr, len, HAL_MAX_DELAY); return len; }
#else
int fputc(int ch, FILE *f) { HAL_UART_Transmit(APP_UART, (uint8_t *)&ch, 1, HAL_MAX_DELAY); return ch; }
#endif

// --- Normalization Constants (CRITICAL: MUST MATCH PYTHON TRAINING) ---
// REPLACE THESE PLACEHOLDERS with the exact constants printed by the Python script
static const float MFCC_MEAN[NUM_FEATURES] = {
    130.1991f, 7.5319f, 0.8439f, -1.0414f, -1.4394f, 
    -1.2013f, -0.8657f, -0.0500f, -0.2905f, -0.0265f, 
    -0.5471f, -0.3123f, -0.4369f, 
}; // 13 elements

static const float MFCC_STD[NUM_FEATURES] = {
    11.8064f, 2.6545f, 2.1868f, 1.7994f, 1.5469f, 
    1.9470f, 1.0526f, 0.9105f, 0.7694f, 0.8158f, 
    0.6365f, 0.4717f, 0.4688f, 
}; // 13 elements


// --- Buffers and Instance ---
int16_t audio_rx_buffer[AUDIO_SAMPLES]; 
float averaged_mfccs[NUM_FEATURES]; 
float normalized_features[NUM_FEATURES];
float frame_mfccs[NUM_FEATURES];

// Global MFCC Instance
MfccInstance* mfcc_instance = NULL;

/**
 * @brief Performs Z-score normalization on the 13 features.
 */
static void normalize_mfccs(float* raw_data, float* out) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        float std = (MFCC_STD[i] < 1e-6f) ? 1e-6f : MFCC_STD[i]; 
        out[i] = (raw_data[i] - MFCC_MEAN[i]) / std;
    }
}

/**
 * @brief Computes the average MFCC vector for the whole audio buffer.
 */
static int calculate_averaged_mfccs(const int16_t* audio_buffer, uint32_t buf_len) {
    if (!mfcc_instance) return -1;

    if (buf_len < FFT_LEN) return -1;
    uint32_t num_frames = (buf_len - FFT_LEN) / HOP_LEN + 1;
    if (num_frames == 0) return -1;

    memset(averaged_mfccs, 0, sizeof(averaged_mfccs));
    
    for (uint32_t i = 0; i < num_frames; i++) {
        const int16_t* frame_ptr = audio_buffer + (i * HOP_LEN);
        mfcc_compute(mfcc_instance, frame_ptr, frame_mfccs);

        for (int k = 0; k < NUM_FEATURES; k++) {
            averaged_mfccs[k] += frame_mfccs[k];
        }
    }

    float inv_num_frames = 1.0f / (float)num_frames;
    for (int k = 0; k < NUM_FEATURES; k++) {
        averaged_mfccs[k] *= inv_num_frames;
    }

    return 0; // Success
}


int app_main(void) {
    printf("\r\n--- Q5: Edge AI Keyword Spotting ---\r\n");

    printf("[MCU] Starting MFCC module initialization (slow operation)...\n");

    // Initialize MFCC Instance. NUM_FEATURES is 13.
    mfcc_instance = mfcc_create_instance(FFT_LEN, NUM_MEL_FILTERS, NUM_FEATURES);
    
    if (!mfcc_instance) {
        printf("FATAL: MFCC Initialization failed.\n");
        return -1;
    }
    
    printf("[MCU] Initialization complete. Starting inference loop.\n");


    while(1) {
        printf(READY_SIGNAL);

        // 1. Receive Raw Audio Buffer (16 KB)
        if (HAL_UART_Receive(APP_UART, (uint8_t*)audio_rx_buffer, sizeof(audio_rx_buffer), 5000) == HAL_OK) {
            
            // 2. Extract Features (MFCC Average)
            int status = calculate_averaged_mfccs(audio_rx_buffer, AUDIO_SAMPLES);

            if (status == 0) {
                
                // 3. Normalize Features
                normalize_mfccs(averaged_mfccs, normalized_features);

                // 4. Predict using the Single Neuron
                float prob = neuron_predict(normalized_features);

                // 5. Send Result (Class 1 is the Target Keyword, Class 0 is Not Target)
                if (prob > 0.5f) {
                    printf("PREDICTION: 1 (Target, Prob: %.2f)\n", prob); 
                } else {
                    printf("PREDICTION: 0 (Other, Prob: %.2f)\n", prob); 
                }
            
            } else {
                printf("Error: Feature extraction failed (Frame calculation error).\n");
            }
        
        } else {
            // Error: UART Receive timeout or error.
        }
    }
}