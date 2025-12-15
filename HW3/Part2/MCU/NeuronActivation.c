#include "main.h"
#include <stdio.h>
#include "motion_features.h" // Your extraction logic
#include "neuron_config.h"   // Generated neuron weights

// --- Configuration ---
#define WINDOW_SIZE 80
#define READY_SIGNAL "[READY]\n"

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

#ifdef __GNUC__
int _write(int file, char *ptr, int len) { HAL_UART_Transmit(APP_UART, (uint8_t*)ptr, len, HAL_MAX_DELAY); return len; }
#else
int fputc(int ch, FILE *f) { HAL_UART_Transmit(APP_UART, (uint8_t *)&ch, 1, HAL_MAX_DELAY); return ch; }
#endif

// Buffers
// 3 axes * 80 samples = 240 floats
float rx_buffer[WINDOW_SIZE * 3]; 
float acc_x[WINDOW_SIZE];
float acc_y[WINDOW_SIZE];
float acc_z[WINDOW_SIZE];

// Feature array for the neuron (Size 10)
float neuron_inputs[NUM_FEATURES];

int app_main(void) {
    printf("\r\n--- Q4: Edge AI HAR (Raw Data) ---\r\n");

    while(1) {
        printf(READY_SIGNAL);

        // 1. Receive Raw Window (80 samples * 3 axes * 4 bytes = 960 bytes)
        if (HAL_UART_Receive(APP_UART, (uint8_t*)rx_buffer, sizeof(rx_buffer), 2000) == HAL_OK) {
            
            // 2. De-interleave / Unpack
            // Assuming Python sends [x0, y0, z0, x1, y1, z1 ...] or [x...][y...][z...]
            // Let's assume Python sends flat X array, then Y, then Z for simplicity.
            // Adjust loop based on Python packing order.
            // Setup below assumes: X[0..79], Y[0..79], Z[0..79]
            for(int i=0; i<WINDOW_SIZE; i++) {
                acc_x[i] = rx_buffer[i];
                acc_y[i] = rx_buffer[WINDOW_SIZE + i];
                acc_z[i] = rx_buffer[WINDOW_SIZE*2 + i];
            }

            // 3. Extract Features (On MCU)
            MotionFeatures feats;
            int status = extract_motion_features(acc_x, acc_y, acc_z, WINDOW_SIZE, &feats);

            if (status == 0) {
                // 4. Map Struct to Linear Array (Order matches Python training)
                neuron_inputs[0] = feats.mean_x;
                neuron_inputs[1] = feats.mean_y;
                neuron_inputs[2] = feats.mean_z;
                neuron_inputs[3] = (float)feats.pos_count_x;
                neuron_inputs[4] = (float)feats.pos_count_y;
                neuron_inputs[5] = (float)feats.pos_count_z;
                neuron_inputs[6] = feats.fft_std_dev_x;
                neuron_inputs[7] = feats.fft_std_dev_y;
                neuron_inputs[8] = feats.fft_std_dev_z;
                neuron_inputs[9] = feats.signal_magnitude_area;

                // 5. Predict
                float prob = neuron_predict(neuron_inputs);

                // 6. Send Result
                if (prob > 0.5f) {
                    printf("PREDICTION: 1 (Prob: %.2f)\n", prob);
                } else {
                    printf("PREDICTION: 0 (Prob: %.2f)\n", prob);
                }
            } else {
                printf("Error: Feature extraction failed.\n");
            }
        
        } else {
            // Flush on timeout
            uint8_t d; while(HAL_UART_Receive(APP_UART, &d, 1, 1) == HAL_OK);
        }
    }
}