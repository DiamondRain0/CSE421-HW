#include "main.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Include feature extraction and model headers
#include "hu.h" // Hu Moments calculation
#include "hu_moment_neuron_config.h" // Neuron configuration

// --- Global Configuration ---
#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_SIZE (IMAGE_ROWS * IMAGE_COLS) // 784 pixels
#define NUM_FEATURES HU_MOMENTS_COUNT // 7

#define READY_SIGNAL "[READY]\n"

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

#ifdef __GNUC__
int _write(int file, char *ptr, int len) { HAL_UART_Transmit(APP_UART, (uint8_t*)ptr, len, HAL_MAX_DELAY); return len; }
#else
int fputc(int ch, FILE *f) { HAL_UART_Transmit(APP_UART, (uint8_t *)&ch, 1, HAL_MAX_DELAY); return ch; }
#endif

// --- Normalization Constants (CRITICAL: MUST MATCH PYTHON TRAINING) ---
static const float HU_MEAN[NUM_FEATURES] = {
    2.7426f, 6.0290f, 6.9939f, 6.9992f, 2.9260f, 
    2.6117f, -1.6582f, 
}; // 7 elements

static const float HU_STD[NUM_FEATURES] = {
    0.1298f, 0.4724f, 0.0140f, 0.0039f, 6.3593f, 
    6.4943f, 6.8005f, 
}; // 7 elements

// --- Buffers ---
// Raw input buffer (8-bit grayscale pixels)
uint8_t image_rx_buffer[IMAGE_SIZE]; 
// Buffer to hold 32-bit floating point pixels for moment calculation
float image_float_buffer[IMAGE_SIZE]; 

// Buffer for raw Hu Moments (output of calculate_hu_moments)
float raw_hu_moments[NUM_FEATURES]; 
// Buffer for normalized features (neuron input)
float normalized_features[NUM_FEATURES];

/**
 * @brief Applies log10 transformation to the Hu moments, matching Python's feature scaling.
 * 
 * Python: hu_moments = -np.sign(hu_moments_raw) * np.log10(np.abs(hu_moments_raw) + 1e-7)
 */
static void apply_log10_transform(float* moments) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        float val = moments[i];
        float abs_val = fabsf(val);
        float sign_val = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        
        // Ensure argument to log10f is positive
        float log_arg = (abs_val < FLT_EPSILON) ? 1e-7f : abs_val;
        
        // Apply the transformation
        moments[i] = -sign_val * log10f(log_arg);
    }
}


/**
 * @brief Performs Z-score normalization on the 7 features.
 */
static void normalize_hu_moments(float* raw_data, float* out) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Apply Z-score: (X - Mean) / StdDev
        float std = (HU_STD[i] < 1e-6f) ? 1e-6f : HU_STD[i]; 
        out[i] = (raw_data[i] - HU_MEAN[i]) / std;
    }
}

int app_main(void) {
    printf("\r\n--- Q6: Edge AI Digit Recognition (Hu Moments) ---\r\n");

    while(1) {
        printf(READY_SIGNAL);

        // 1. Receive Raw Image Buffer (784 bytes)
        if (HAL_UART_Receive(APP_UART, (uint8_t*)image_rx_buffer, sizeof(image_rx_buffer), 5000) == HAL_OK) {
            
            // 2. Prepare Pixels (Convert uin8_t to float for moment calculation)
            for (int i = 0; i < IMAGE_SIZE; i++) {
                // Pixels are typically 0 (black) to 255 (white).
                // Hu moments usually work on intensity values directly.
                image_float_buffer[i] = (float)image_rx_buffer[i];
            }
            
            // 3. Calculate Hu Moments
            int status = calculate_hu_moments(
                image_float_buffer, 
                IMAGE_ROWS, 
                IMAGE_COLS, 
                raw_hu_moments
            );

            if (status == 0) {
                
                // 4. Apply Log Transformation (Rotation and Scale Invariance)
                apply_log10_transform(raw_hu_moments);
                
                // 5. Apply Z-score Normalization
                normalize_hu_moments(raw_hu_moments, normalized_features);

                // 6. Predict using the Single Neuron (0 vs Not 0)
                float prob = neuron_predict(normalized_features);

                // 7. Send Result (Class 1 is '0', Class 0 is 'Not 0')
                if (prob > 0.5f) {
                    printf("PREDICTION: 1 (Digit 0, Prob: %.2f)\n", prob); 
                } else {
                    printf("PREDICTION: 0 (Digit Not 0, Prob: %.2f)\n", prob); 
                }
            
            } else if (status == -1) {
                printf("Error: Image was empty (all pixels 0).\n");
            } else {
                printf("Error: Hu moment calculation failed.\n");
            }
        
        } else {
            // Handle UART timeout/error during receive
            uint8_t d; while(HAL_UART_Receive(APP_UART, &d, 1, 1) == HAL_OK);
        }
    }
}