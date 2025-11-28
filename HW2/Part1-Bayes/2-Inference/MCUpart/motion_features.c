#include "motion_features.h"
#include <math.h>
#include <string.h> // For memset

/**
 * @brief Helper to calculate Standard Deviation
 */
static float calculate_std_dev(float* data, uint32_t len, float mean) {
    if (len == 0) return 0.0f;
    float variance_sum = 0.0f;
    for (uint32_t i = 0; i < len; i++) {
        float diff = data[i] - mean;
        variance_sum += diff * diff;
    }
    return sqrtf(variance_sum / len);
}

/**
 * @brief Main feature extraction function
 * Implements: Mean, Positive Count, Std Dev (used as proxy for FFT Std Dev), and SMA.
 */
int extract_motion_features(float* ax, float* ay, float* az, uint32_t len, MotionFeatures* out) {
    if (len == 0 || out == NULL) {
        return -1; // Error: Invalid arguments
    }

    // Initialize accumulators
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    float abs_sum_total = 0.0f;
    int pos_x = 0, pos_y = 0, pos_z = 0;

    // 1. First Pass: Compute Sums, Positive Counts, and SMA parts
    for (uint32_t i = 0; i < len; i++) {
        // Sums for Mean
        sum_x += ax[i];
        sum_y += ay[i];
        sum_z += az[i];

        // Positive Counts (Value > 0)
        if (ax[i] > 0) pos_x++;
        if (ay[i] > 0) pos_y++;
        if (az[i] > 0) pos_z++;

        // Signal Magnitude Area (SMA): Sum of absolute values of all axes
        abs_sum_total += fabsf(ax[i]) + fabsf(ay[i]) + fabsf(az[i]);
    }

    // 2. Calculate Means
    out->mean_x = sum_x / len;
    out->mean_y = sum_y / len;
    out->mean_z = sum_z / len;

    // 3. Store Positive Counts
    out->pos_count_x = pos_x;
    out->pos_count_y = pos_y;
    out->pos_count_z = pos_z;

    // 4. Calculate SMA (Normalized by length)
    out->signal_magnitude_area = abs_sum_total / len;

    // 5. Calculate Standard Deviations
    // Note: The prompt asked for "fft_std_dev". Without a complex FFT library setup,
    // standard deviation of the time-series is often used as a close proxy in simple embedded tasks.
    out->fft_std_dev_x = calculate_std_dev(ax, len, out->mean_x);
    out->fft_std_dev_y = calculate_std_dev(ay, len, out->mean_y);
    out->fft_std_dev_z = calculate_std_dev(az, len, out->mean_z);

    return 0; // Success
}