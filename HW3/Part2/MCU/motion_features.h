#ifndef MOTION_FEATURES_H
#define MOTION_FEATURES_H

#include <stdint.h>

// Structure to hold the extracted features
// MUST Match the feature order expected by your Bayes classifier!
typedef struct {
    float mean_x;
    float mean_y;
    float mean_z;
    int pos_count_x;
    int pos_count_y;
    int pos_count_z;
    float fft_std_dev_x;
    float fft_std_dev_y;
    float fft_std_dev_z;
    float signal_magnitude_area;
} MotionFeatures;

// Function Prototype
int extract_motion_features(float* ax, float* ay, float* az, uint32_t len, MotionFeatures* out);

#endif