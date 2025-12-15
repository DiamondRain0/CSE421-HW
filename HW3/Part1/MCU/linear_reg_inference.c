#include "linear_reg_inference.h"
#include "linear_reg_config.h" // Contains COEFFS and OFFSET

/*---------------------------------------------------------------------------
 * Linear Regression Prediction Logic
 * Calculates y = b + w1*x1 + w2*x2 ...
 *---------------------------------------------------------------------------*/
float linear_reg_predict(float *features) {
  float result = OFFSET;

  for (int i = 0; i < NUM_FEATURES; i++) {
    result += features[i] * COEFFS[i];
  }

  return result;
}