/*---------------------------------------------------------------------------
 * Includes
 *---------------------------------------------------------------------------*/
#include "main.h"
#include "linear_reg_inference.h"
#include "linear_reg_config.h" // Defines NUM_FEATURES (e.g., 5)
//#include <math.h>
#include <stdint.h>
#include <stdio.h>
//#include <stdlib.h>
#include <string.h>

/*---------------------------------------------------------------------------
 * Configuration & Protocols
 *---------------------------------------------------------------------------*/
#define MAX_WINDOW_SIZE 20 // We only need 5, but keeping it safe

// Protocol Signals (Matches your reference exactly)
#define INITIALIZED_SIGNAL "[INITIALIZED]\n"
#define CONFIG_OK_SIGNAL "[CONFIG_OK]\n"
#define READY_SIGNAL "[READY]\n"
#define SAMPLE_OK_SIGNAL "[SAMPLE_OK]\n"
#define PROCESS_COMMAND "[PROCESS]\n"
#define END_MARKER "[END_OF_PREDICTION]\n"

/*---------------------------------------------------------------------------
 * Global Handles & Buffers
 *---------------------------------------------------------------------------*/
extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

// Data Buffer for Temperature History
static float input_features_buffer[MAX_WINDOW_SIZE];

// UART Receiving Structures
// We receive 1 float (temperature) at a time
typedef struct {
  float value;
} Sample;

typedef union {
  Sample sample;
  char command[sizeof(Sample)]; // Ensure union is large enough for command checking
} RxPayload;

typedef uint32_t ConfigHeader;

/*---------------------------------------------------------------------------
 * UART Helper Functions (Printf Redirection)
 *---------------------------------------------------------------------------*/
#ifdef __GNUC__
int _write(int file, char *ptr, int len) {
  HAL_UART_Transmit(APP_UART, (uint8_t *)ptr, len, HAL_MAX_DELAY);
  return len;
}
#else
int fputc(int ch, FILE *f) {
  HAL_UART_Transmit(APP_UART, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
  return ch;
}
#endif

void flush_uart_rx_buffer(void) {
  uint8_t d;
  while (HAL_UART_Receive(APP_UART, &d, 1, 10) == HAL_OK)
    ;
}

/*---------------------------------------------------------------------------
 * Processing Function
 * 1. Runs Regression
 * 2. Sends Result to PC
 *---------------------------------------------------------------------------*/
void compute_prediction_and_send(void) {
  // 1. Run Inference
  // We pass the buffer containing the received features (t-5, t-4...)
  float prediction = linear_reg_predict(input_features_buffer);

  // 2. Send Prediction to Python
  // Format: "PREDICTED_VALUE: <Float>"
  printf("PREDICTED_VALUE: %.4f\n", prediction);

  // 3. End Marker
  printf(END_MARKER);
}

/*---------------------------------------------------------------------------
 * Main Application
 * Copy this content into your main() or call app_main() from main()
 *---------------------------------------------------------------------------*/
int app_main(void) {
  uint32_t configured_window_size = 0;

  // --- PHASE 1: CONFIGURATION ---
  // Wait for PC to send the number of features (should be 5)
  while (configured_window_size == 0) {
    printf(INITIALIZED_SIGNAL);
    ConfigHeader received_size;

    // Blocking receive with timeout (1000ms)
    if (HAL_UART_Receive(APP_UART, (uint8_t *)&received_size,
                         sizeof(ConfigHeader), 1000) != HAL_OK) {
      continue;
    }

    // Validation
    if (received_size > 0 && received_size <= MAX_WINDOW_SIZE) {
        // Optional: Check if received_size matches NUM_FEATURES from config.h
        if(received_size == NUM_FEATURES) {
            configured_window_size = received_size;
            printf("Config: Features = %lu\n", configured_window_size);
            printf(CONFIG_OK_SIGNAL);
        } else {
             printf("Error: Expected %d features, got %lu\n", NUM_FEATURES, received_size);
        }
    } else {
      printf("Error: Invalid Size %lu\n", received_size);
    }
  }

  // --- PHASE 2: PROCESSING LOOP ---
  uint32_t current_sample_index = 0;
  RxPayload rx_payload;

  for (;;) {
    flush_uart_rx_buffer();
    current_sample_index = 0;
    printf(READY_SIGNAL); // Tell PC we are ready for a new batch

    // Fill the buffer with 'configured_window_size' samples
    while (current_sample_index < configured_window_size) {
      
      // Receive 1 Sample (4 bytes)
      if (HAL_UART_Receive(APP_UART, (uint8_t *)&rx_payload, sizeof(RxPayload),
                           5000) != HAL_OK) {
        printf("Timeout waiting for data. Resetting...\n");
        break;
      }

      // Check if it's a command (optional safety, similar to reference)
      if (strncmp(rx_payload.command, PROCESS_COMMAND,
                  strlen(PROCESS_COMMAND)) == 0) {
         // Logic to handle early exit if needed, usually not for regression
         break;
      }

      // Store Data
      input_features_buffer[current_sample_index] = rx_payload.sample.value;
      current_sample_index++;

      // Acknowledge receipt
      printf(SAMPLE_OK_SIGNAL);
    }

    // Buffer full? Process it.
    if (current_sample_index == configured_window_size) {
      compute_prediction_and_send();
    }
  }
}