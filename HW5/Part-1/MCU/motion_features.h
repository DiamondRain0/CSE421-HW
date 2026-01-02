#ifndef MOTION_FEATURES_H
#define MOTION_FEATURES_H

#include <stdint.h>

typedef struct {
    float mean_x, mean_y, mean_z;    // 0, 1, 2
    float std_x,  std_y,  std_z;     // 3, 4, 5
    float max_x,  max_y,  max_z;     // 6, 7, 8
    float min_x,  min_y,  min_z;     // 9, 10, 11
    float sma;                       // 12
} MotionFeatures;



void extract_motion_features(
    float* ax, float* ay, float* az,
    uint32_t len,
    MotionFeatures* out
);

#endif