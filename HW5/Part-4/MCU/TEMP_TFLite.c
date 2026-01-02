#include "main.h"
#include <stdio.h>
#include <string.h>

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

// From temp_inference.cpp
float run_regression_inference(float* input_window);

// Sliding window: 4 time-steps, each with 22 features (88 floats total)
float feature_window[4][22];
float newest_reading[22];

int app_main(void) {
    printf("--- Temperature Prediction System (App 12.10) ---\n");
    
    // 1. Initialize sliding window to zero
    memset(feature_window, 0, sizeof(feature_window));

    while (1) {
        // 2. Send synchronization signal to the Python script
        printf("[READY]\n");

        // 3. Wait for exactly 22 floats (88 bytes) from the PC
        // Note: (uint8_t*) cast is required by the HAL function
        if (HAL_UART_Receive(APP_UART, (uint8_t*)newest_reading, 22 * sizeof(float), HAL_MAX_DELAY) == HAL_OK) {
            
            // 4. Sliding Window Update:
            // Shift indices [1, 2, 3] into positions [0, 1, 2]
            memmove(&feature_window[0], &feature_window[1], 3 * 22 * sizeof(float));
            
            // Copy the new reading into index [3]
            memcpy(&feature_window[3], newest_reading, 22 * sizeof(float));
            
            // 5. Run Inference on the flattened 88-float window
            float prediction = run_regression_inference((float*)feature_window);
            
            // 6. Print the result for the Python script to parse
            // Format must match: "PREDICTED TEMPERATURE: %f"
            printf("PREDICTED TEMPERATURE: %.2f C\n", prediction);
        }
    }
}