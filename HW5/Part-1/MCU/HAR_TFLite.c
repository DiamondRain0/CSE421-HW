#include "har_inference.h"
#include "motion_features.h"
#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// --- SETTINGS ---
#define MAX_SAMPLES 256

extern UART_HandleTypeDef huart1;
#define APP_UART &huart1

float ax[MAX_SAMPLES];
float ay[MAX_SAMPLES];
float az[MAX_SAMPLES];

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
  while (HAL_UART_Receive(APP_UART, &d, 1, 10) == HAL_OK);
}

int app_main(void) {
  HAL_Delay(500); // Wait for UART stable
  printf("\r\n--- HAR Motion Classifier ---\r\n");

  // 1. Setup Model
  har_init();
  printf("[INITIALIZED]\r\n");

  while (1) {
    printf("[READY]\r\n");

    // 2. Receive window size
    int32_t n = 0;
    if (HAL_UART_Receive(APP_UART, (uint8_t *)&n, 4, HAL_MAX_DELAY) != HAL_OK)
      continue;

    if (n > MAX_SAMPLES || n <= 0) {
      printf("Error: Invalid Window\r\n");
      flush_uart();
      continue;
    }

    // 3. Receive samples (x,y,z floats)
    HAL_UART_Receive(APP_UART, (uint8_t *)ax, n * 4, 5000);
    HAL_UART_Receive(APP_UART, (uint8_t *)ay, n * 4, 5000);
    HAL_UART_Receive(APP_UART, (uint8_t *)az, n * 4, 5000);

    printf("DEBUG_IN: x[0]=%f, y[0]=%f, z[0]=%f\r\n", ax[0], ay[0], az[0]);

    // 4. Feature extraction
    MotionFeatures f;
    extract_motion_features(ax, ay, az, (int32_t)n, &f);
    printf("DEBUG_FEAT: MeanX=%f, SMA=%f\r\n", f.mean_x, f.sma);

    // 5. Map struct to array in EXACT order of Python training:
    // np.concatenate([means, stds, maxs, mins, [sma]])
    float feature_vec[13] = {
        f.mean_x, f.mean_y, f.mean_z, // 0, 1, 2
        f.std_x,  f.std_y,  f.std_z,  // 3, 4, 5
        f.max_x,  f.max_y,  f.max_z,  // 6, 7, 8
        f.min_x,  f.min_y,  f.min_z,  // 9, 10, 11
        f.sma                         // 12
    };

    // 6. Prediction
    int pred = har_predict(feature_vec);
    printf("PREDICTED_CLASS: %d\r\n", pred);
  }
}