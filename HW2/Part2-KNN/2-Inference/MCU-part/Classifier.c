#include "knn_inference.h"   // kNN prediction logic
#include "knn_mfcc_config.h" // The model trained by Python
#include "main.h"
#include "mfcc.h" // Your MFCC library
#include <stdio.h>
#include <stdlib.h>

// --- SETTINGS ---
#define SAMPLE_RATE 8000
#define FFT_LEN 1024
#define HOP_LEN 512
#define NUM_MEL_FILTERS 20
#define NUM_DCT_COEFFS 13
#define MAX_AUDIO_SAMPLES 10000

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

// Large buffer for incoming audio
int16_t raw_audio_buffer[MAX_AUDIO_SAMPLES];
MfccInstance *mfcc_inst = NULL;

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

void flush_uart(void) {
  uint8_t d;
  while (HAL_UART_Receive(APP_UART, &d, 1, 1) == HAL_OK)
    ;
}

int app_main(void) {
  printf("\r\n--- FSDD Audio Classifier ---\r\n");

  // 1. Setup MFCC
  mfcc_inst = mfcc_create_instance(FFT_LEN, NUM_MEL_FILTERS, NUM_DCT_COEFFS);
  if (!mfcc_inst) {
    printf("Error: Heap allocation failed.\r\n");
    return -1;
  }
  printf("[INITIALIZED]\r\n");

  while (1) {
    // 2. Handshake
    printf("[READY]\r\n");

    // 3. Receive Size (4 bytes)
    uint32_t num_samples = 0;
    if (HAL_UART_Receive(APP_UART, (uint8_t *)&num_samples, 4, 2000) !=
        HAL_OK) {
      continue;
    }

    // Safety check
    if (num_samples > MAX_AUDIO_SAMPLES) {
      printf("Error: Audio too long (%lu)\r\n", num_samples);
      flush_uart();
      continue;
    }

    // 4. Receive Raw Audio (PCM Int16)
    // We assume PC sends Little Endian int16
    if (HAL_UART_Receive(APP_UART, (uint8_t *)raw_audio_buffer, num_samples * 2,
                         5000) != HAL_OK) {
      printf("Error: Audio Timeout\r\n");
      continue;
    }

    // 5. Processing (Edge AI)
    float avg_mfcc[NUM_DCT_COEFFS] = {0};
    float temp_mfcc[NUM_DCT_COEFFS];
    int frame_count = 0;

    // Sliding window over audio
    for (uint32_t i = 0; i <= (num_samples - FFT_LEN); i += HOP_LEN) {
      mfcc_compute(mfcc_inst, &raw_audio_buffer[i], temp_mfcc);

      // Accumulate
      for (int k = 0; k < NUM_DCT_COEFFS; k++) {
        avg_mfcc[k] += temp_mfcc[k];
      }
      frame_count++;
    }

    // Average
    if (frame_count > 0) {
      for (int k = 0; k < NUM_DCT_COEFFS; k++) {
        avg_mfcc[k] /= (float)frame_count;
      }
    }

    // 6. Prediction
    int pred_index = knn_predict(avg_mfcc);

    // 7. Result
    if (pred_index >= 0 && pred_index < NUM_UNIQUE_CLASSES) {
      printf("PREDICTED_CLASS: %s\r\n", CLASS_NAMES[pred_index]);
    } else {
      printf("PREDICTED_CLASS: ?\r\n");
    }
  }
}