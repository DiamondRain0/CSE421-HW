#include "knn_inference.h"
#include <float.h>
#include <math.h>

typedef struct {
  float dist_sq;
  int label;
} Neighbor;

int knn_predict(float *features) {
  Neighbor best[NUM_NEIGHBORS];

  // Init with max distance
  for (int i = 0; i < NUM_NEIGHBORS; i++) {
    best[i].dist_sq = FLT_MAX;
    best[i].label = -1;
  }

  // Scan all training samples stored in Flash (DATA array)
  for (int i = 0; i < NUM_SAMPLES; i++) {
    float d = 0.0f;
    for (int j = 0; j < NUM_FEATURES; j++) {
      float diff = features[j] - DATA[i][j];
      d += diff * diff;
    }

    // Check against current bests
    // (Simple insertion sort logic for top-k)
    for (int k = 0; k < NUM_NEIGHBORS; k++) {
      if (d < best[k].dist_sq) {
        // Shift others down
        for (int m = NUM_NEIGHBORS - 1; m > k; m--) {
          best[m] = best[m - 1];
        }
        // Insert new best
        best[k].dist_sq = d;
        best[k].label = DATA_LABELS[i];
        break;
      }
    }
  }

  // Majority Vote
  int votes[NUM_CLASSES] = {0};
  for (int i = 0; i < NUM_NEIGHBORS; i++) {
    if (best[i].label != -1)
      votes[best[i].label]++;
  }

  int winner = -1;
  int max_v = -1;
  for (int c = 0; c < NUM_CLASSES; c++) {
    if (votes[c] > max_v) {
      max_v = votes[c];
      winner = c;
    }
  }
  return winner;
}