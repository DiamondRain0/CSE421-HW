#include "hu.h" // Your provided file
#include "main.h"
#include "svm_logic.h" // SVM Prediction
#include <math.h>
#include <stdio.h>

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

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

// Constants
#define IMG_ROWS 28
#define IMG_COLS 28
#define IMG_SIZE (IMG_ROWS * IMG_COLS)
#define READY_SIGNAL "[READY]\n"

// Buffers
uint8_t rx_image_buffer[IMG_SIZE];  // Receives 784 bytes (0-255)
float image_float_buffer[IMG_SIZE]; // Converted for Hu calc
float hu_moments[HU_MOMENTS_COUNT]; // Stores result

int app_main(void) {
  printf("\r\n--- Q3: End-to-End Digit Rec ---\r\n");
  printf("Send raw 28x28 bytes (784 bytes)...\r\n");

  while (1) {
    printf(READY_SIGNAL);

    // 1. Receive Raw Image (784 bytes) via UART
    if (HAL_UART_Receive(APP_UART, rx_image_buffer, IMG_SIZE, 3000) == HAL_OK) {

      // 2. Pre-process (Threshold & Float Conversion)
      // Python trained on (val > 128) -> 1.0
      for (int i = 0; i < IMG_SIZE; i++) {
        image_float_buffer[i] = (rx_image_buffer[i] > 128) ? 1.0f : 0.0f;
      }

      // 3. Calculate Hu Moments (Using YOUR hu.c)
      int status = calculate_hu_moments(image_float_buffer, IMG_ROWS, IMG_COLS,
                                        hu_moments);

      if (status == 0) {
        // Log Transform
        for (int i = 0; i < HU_MOMENTS_COUNT; i++) {
          float val = hu_moments[i];
          float abs_val = fabsf(val) + 1e-6f;
          float sign = (val >= 0) ? 1.0f : -1.0f;
          hu_moments[i] = -1.0f * sign * log10f(abs_val);
        }

        // --- DEBUG: Print Features to verify with Python ---
        // Uncomment this once to check values in Serial Monitor
        // printf("DEBUG_FEAT: %.4f %.4f %.4f\n", hu_moments[0], hu_moments[1],
        // hu_moments[2]);

        int digit = svm_predict_digit(hu_moments);
        printf("PREDICTED_CLASS: %d\n", digit);

      } else {
        printf("Error: Hu calc failed (code %d)\n", status);
      }

    } else {
      // Flush on timeout
      uint8_t d;
      while (HAL_UART_Receive(APP_UART, &d, 1, 1) == HAL_OK)
        ;
    }
  }
}