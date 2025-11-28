#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "motion_features.h"

// --- Configuration ---
#define MAX_WINDOW_SIZE 256 // The absolute maximum buffer size

// --- Protocol Signals ---
#define INITIALIZED_SIGNAL       "[INITIALIZED]\n"
#define CONFIG_OK_SIGNAL         "[CONFIG_OK]\n"
#define READY_SIGNAL             "[READY]\n"
#define SAMPLE_OK_SIGNAL         "[SAMPLE_OK]\n"
#define PROCESS_COMMAND          "[PROCESS]\n"
#define END_MARKER               "[END_OF_FEATURES]\n"

// --- Global Handles and Buffers ---
extern UART_HandleTypeDef huart1;
#define APP_UART &huart1
static float sensor_x_buffer[MAX_WINDOW_SIZE];
static float sensor_y_buffer[MAX_WINDOW_SIZE];
static float sensor_z_buffer[MAX_WINDOW_SIZE];

// (printf retargeting and flush_uart_rx_buffer functions remain unchanged)
#ifdef __GNUC__
int _write(int file, char *ptr, int len) { HAL_UART_Transmit(APP_UART, (uint8_t*)ptr, len, HAL_MAX_DELAY); return len; }
#else
int fputc(int ch, FILE *f) { HAL_UART_Transmit(APP_UART, (uint8_t *)&ch, 1, HAL_MAX_DELAY); return ch; }
#endif
void flush_uart_rx_buffer(void) { uint8_t d; while (HAL_UART_Receive(APP_UART, &d, 1, 10) == HAL_OK); }

// Data structures for receiving data
typedef struct { float x, y, z; } Sample;
typedef union { Sample sample; char command[sizeof(Sample)]; } RxPayload;
typedef uint32_t ConfigHeader;

// (compute_motion_and_send is updated to take window_size as an argument again)
void compute_motion_and_send(uint32_t window_size) {
    printf("Starting motion feature computation for %lu samples...\n", window_size);
    MotionFeatures features_output;
    int status = extract_motion_features(sensor_x_buffer, sensor_y_buffer, sensor_z_buffer, window_size, &features_output);
    if (status != 0) { printf("Error: Motion feature extraction failed with code %d\n", status); }
    else {
        printf("Computation complete. Sending results:\n");
        printf("MeanX: %f\n", features_output.mean_x); printf("MeanY: %f\n", features_output.mean_y); printf("MeanZ: %f\n", features_output.mean_z);
        printf("PosCountX: %d\n", features_output.pos_count_x); printf("PosCountY: %d\n", features_output.pos_count_y); printf("PosCountZ: %d\n", features_output.pos_count_z);
        printf("FFTStdDevX: %f\n", features_output.fft_std_dev_x); printf("FFTStdDevY: %f\n", features_output.fft_std_dev_y); printf("FFTStdDevZ: %f\n", features_output.fft_std_dev_z);
        printf("SignalMagArea: %f\n", features_output.signal_magnitude_area);
    }
    printf(END_MARKER);
}

/* ======================================================
            MAIN APP LOOP (Configurable Real-time version)
   ====================================================== */
int app_main(void) {
    uint32_t configured_window_size = 0;
    
    // --- PHASE 1: CONFIGURATION ---
    // This loop ensures we get a valid configuration before proceeding.
    while(configured_window_size == 0) {
        printf(INITIALIZED_SIGNAL);
        ConfigHeader received_size;
        if (HAL_UART_Receive(APP_UART, (uint8_t*)&received_size, sizeof(ConfigHeader), HAL_MAX_DELAY) != HAL_OK) {
            continue; // If timeout or error, just try again
        }
        
        if (received_size > 0 && received_size <= MAX_WINDOW_SIZE) {
            configured_window_size = received_size;
            printf("Configuration received: Window size set to %lu\n", configured_window_size);
            printf(CONFIG_OK_SIGNAL);
        } else {
            printf("Error: Invalid window size %lu received. Must be > 0 and <= %d.\n", received_size, MAX_WINDOW_SIZE);
        }
    }

    // --- PHASE 2: PROCESSING ---
    uint32_t current_sample_index = 0;
    RxPayload rx_payload;

    for (;;) { // This outer loop represents a full window cycle
        flush_uart_rx_buffer();
        current_sample_index = 0;
        printf(READY_SIGNAL);

        while (current_sample_index < configured_window_size) {
            if (HAL_UART_Receive(APP_UART, (uint8_t*)&rx_payload, sizeof(RxPayload), 15000) != HAL_OK) {
                printf("Info: Timed out waiting for sample. Resetting window.\n");
                break; // Break inner loop to restart the window cycle
            }

            if (strncmp(rx_payload.command, PROCESS_COMMAND, strlen(PROCESS_COMMAND)) == 0) {
                 if (current_sample_index > 0) {
                    printf("Process command received. Processing %lu samples.\n", current_sample_index);
                    compute_motion_and_send(current_sample_index); // Process partial window
                 } else {
                    printf("Info: Received PROCESS on empty buffer. Ignoring.\n");
                 }
                break; 
            }

            sensor_x_buffer[current_sample_index] = rx_payload.sample.x;
            sensor_y_buffer[current_sample_index] = rx_payload.sample.y;
            sensor_z_buffer[current_sample_index] = rx_payload.sample.z;
            current_sample_index++;
            printf(SAMPLE_OK_SIGNAL);
        }

        if (current_sample_index == configured_window_size) {
            compute_motion_and_send(configured_window_size);
        }
    }
}
