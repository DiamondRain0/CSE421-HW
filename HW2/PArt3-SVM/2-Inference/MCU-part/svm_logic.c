#include "svm_logic.h"
#include <math.h>
#include <string.h>

// Helper: RBF Kernel
static float rbf_kernel(const float *x, const float *y) {
  float dist_sq = 0.0f;
  for (int i = 0; i < NUM_FEATURES; i++) {
    float d = x[i] - y[i];
    dist_sq += d * d;
  }
  return expf(-svm_gamma * dist_sq);
}

int svm_predict_digit(float *raw_features) {
  // 1. SCALING (StandardScaler)
  float features[NUM_FEATURES];
  for (int i = 0; i < NUM_FEATURES; i++) {
    // Safety: If scale is 0 or garbage, avoid division by zero
    if (SCALER_SCALE[i] != 0.0f) {
      features[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    } else {
      features[i] = 0.0f;
    }
  }

  // 2. Kernel Calc (Using STATIC to save stack memory)
  // IMPORTANT: If NUM_SV is > 1000, this array is too big for stack.
  // We make it static so it lives in RAM.
  static float k_values[NUM_SV];
  for (int i = 0; i < NUM_SV; i++) {
    k_values[i] = rbf_kernel(features, SV[i]);
  }

  // 3. Voting
  int votes[NUM_CLASSES];
  memset(votes, 0, sizeof(votes));

  int k = 0; // Intercept index

  // Reconstruct start indices if not provided
  int start[NUM_CLASSES];
  start[0] = 0;
  for (int i = 1; i < NUM_CLASSES; i++)
    start[i] = start[i - 1] + n_support[i - 1];

  for (int i = 0; i < NUM_CLASSES; i++) {
    for (int j = i + 1; j < NUM_CLASSES; j++) {
      float sum = 0.0f;
      int si = start[i], ci = n_support[i];
      int sj = start[j], cj = n_support[j];

      for (int m = 0; m < ci; m++)
        sum += coeffs[j - 1][si + m] * k_values[si + m];
      for (int m = 0; m < cj; m++)
        sum += coeffs[i][sj + m] * k_values[sj + m];

      // --- FIX IS HERE: Change -= to += ---
      // Scikit-learn exports 'intercept_' which corresponds to +b in (wx + b)
      sum += intercepts[k++];
      // ------------------------------------

      if (sum > 0)
        votes[i]++;
      else
        votes[j]++;
    }
  }

  int winner = 0, max_votes = -1;
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (votes[i] > max_votes) {
      max_votes = votes[i];
      winner = i;
    }
  }
  return winner;
}