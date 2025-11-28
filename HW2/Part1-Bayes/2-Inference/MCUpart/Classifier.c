/*---------------------------------------------------------------------------
 * Includes
 *---------------------------------------------------------------------------*/
#include "main.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// User Libraries
#include "bayes_cls_config.h" // For Naive Bayes arrays (MEANS, etc.)
#include "motion_features.h"  // For extract_motion_features

/*---------------------------------------------------------------------------
 * Configuration & Protocols
 *---------------------------------------------------------------------------*/
#define MAX_WINDOW_SIZE 256

// Protocol Signals
#define INITIALIZED_SIGNAL "[INITIALIZED]\n"
#define CONFIG_OK_SIGNAL "[CONFIG_OK]\n"
#define READY_SIGNAL "[READY]\n"
#define SAMPLE_OK_SIGNAL "[SAMPLE_OK]\n"
#define PROCESS_COMMAND "[PROCESS]\n"
#define END_MARKER "[END_OF_FEATURES]\n"

/*---------------------------------------------------------------------------
 * Global Handles & Buffers
 *---------------------------------------------------------------------------*/
extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

// Data Buffers for Raw Accelerometer
static float sensor_x_buffer[MAX_WINDOW_SIZE];
static float sensor_y_buffer[MAX_WINDOW_SIZE];
static float sensor_z_buffer[MAX_WINDOW_SIZE];

// UART Receiving Structures
typedef struct {
  float x, y, z;
} Sample;
typedef union {
  Sample sample;
  char command[sizeof(Sample)];
} RxPayload;
typedef uint32_t ConfigHeader;

/*---------------------------------------------------------------------------
 * Class Names Mapping
 * (Must match the order in your Python training script)
 *---------------------------------------------------------------------------*/
const char *ACTIVITY_NAMES[] = {
    "Downstairs", // 0
    "Jogging",    // 1
    "Sitting",    // 2
    "Standing",   // 3
    "Upstairs",   // 4
    "Walking"     // 5
};

/*---------------------------------------------------------------------------
 * UART Helper Functions
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
 * Bayes Prediction Logic
 *---------------------------------------------------------------------------*/
int bayes_predict(float *features) {
  int best_class = -1;
  float max_log_prob = -1e30f;

  for (int c = 0; c < NUM_CLASSES; c++) {
    float log_prob = (CLASS_PRIORS[c] > 0.0f) ? logf(CLASS_PRIORS[c]) : -100.0f;

    if (DETS[c] > 0.0f)
      log_prob -= 0.5f * logf(DETS[c]);

    float dist_sum = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
      float diff_i = features[i] - MEANS[c][i];
      for (int j = 0; j < NUM_FEATURES; j++) {
        float diff_j = features[j] - MEANS[c][j];
        dist_sum += diff_i * INV_COVS[c][i][j] * diff_j;
      }
    }
    log_prob -= 0.5f * dist_sum;

    if (log_prob > max_log_prob) {
      max_log_prob = log_prob;
      best_class = c;
    }
  }
  return best_class;
}

/*---------------------------------------------------------------------------
 * Processing Function
 * 1. Extracts Features
 * 2. Predicts Class
 * 3. Sends Result to PC
 *---------------------------------------------------------------------------*/
#define END_MARKER                                                             \
  "[END_OF_PREDICTION]\n" // Update this definition at the top of main.c

void compute_motion_and_send(uint32_t window_size) {
  // Optional: Print status to verify it's running, but strictly not needed for
  // data printf("Processing...\n");

  MotionFeatures feat;

  // 1. Extract Features (Silent)
  int status = extract_motion_features(sensor_x_buffer, sensor_y_buffer,
                                       sensor_z_buffer, window_size, &feat);

  if (status != 0) {
    printf("Error: Feature extraction failed (Code %d)\n", status);
    printf(END_MARKER);
    return;
  }

  // 2. Prepare Input for Classifier
  float ml_input[10];
  ml_input[0] = feat.mean_x;
  ml_input[1] = feat.mean_y;
  ml_input[2] = feat.mean_z;
  ml_input[3] = (float)feat.pos_count_x;
  ml_input[4] = (float)feat.pos_count_y;
  ml_input[5] = (float)feat.pos_count_z;
  ml_input[6] = feat.fft_std_dev_x;
  ml_input[7] = feat.fft_std_dev_y;
  ml_input[8] = feat.fft_std_dev_z;
  ml_input[9] = feat.signal_magnitude_area;

  // 3. Run Classification
  int predicted_class = bayes_predict(ml_input);

  // 4. Send Prediction to Python
  // Format: "PREDICTED_CLASS: <ID>"
  if (predicted_class >= 0 && predicted_class < NUM_CLASSES) {
    printf("PREDICTED_CLASS: %d\n", predicted_class);
  } else {
    printf("PREDICTED_CLASS: -1\n"); // Error code
  }

  // 5. End Marker
  printf(END_MARKER);
}

/*---------------------------------------------------------------------------
 * Main Application
 *---------------------------------------------------------------------------*/
int app_main(void) {
  uint32_t configured_window_size = 0;

  // --- PHASE 1: CONFIGURATION ---
  // Wait for PC to send the window size (e.g., 80)
  while (configured_window_size == 0) {
    printf(INITIALIZED_SIGNAL);
    ConfigHeader received_size;

    // Blocking receive with timeout
    if (HAL_UART_Receive(APP_UART, (uint8_t *)&received_size,
                         sizeof(ConfigHeader), 1000) != HAL_OK) {
      continue;
    }

    if (received_size > 0 && received_size <= MAX_WINDOW_SIZE) {
      configured_window_size = received_size;
      printf("Config: Window Size = %lu\n", configured_window_size);
      printf(CONFIG_OK_SIGNAL);
    } else {
      printf("Error: Invalid Window Size %lu\n", received_size);
    }
  }

  // --- PHASE 2: PROCESSING LOOP ---
  uint32_t current_sample_index = 0;
  RxPayload rx_payload;

  for (;;) {
    flush_uart_rx_buffer();
    current_sample_index = 0;
    printf(READY_SIGNAL); // Tell PC we are ready for a new batch

    // Fill one window
    while (current_sample_index < configured_window_size) {
      // Receive 1 sample (float x, y, z) OR a Command String
      if (HAL_UART_Receive(APP_UART, (uint8_t *)&rx_payload, sizeof(RxPayload),
                           5000) != HAL_OK) {
        printf("Timeout waiting for data. Resetting...\n");
        break;
      }

      // Check if it's a command instead of data
      if (strncmp(rx_payload.command, PROCESS_COMMAND,
                  strlen(PROCESS_COMMAND)) == 0) {
        if (current_sample_index > 0) {
          // Early process command
          compute_motion_and_send(current_sample_index);
        }
        break; // Break inner loop to restart
      }

      // Store Data
      sensor_x_buffer[current_sample_index] = rx_payload.sample.x;
      sensor_y_buffer[current_sample_index] = rx_payload.sample.y;
      sensor_z_buffer[current_sample_index] = rx_payload.sample.z;
      current_sample_index++;

      // Acknowledge receipt
      printf(SAMPLE_OK_SIGNAL);
    }

    // Buffer full? Process it.
    if (current_sample_index == configured_window_size) {
      compute_motion_and_send(configured_window_size);
    }
  }
}