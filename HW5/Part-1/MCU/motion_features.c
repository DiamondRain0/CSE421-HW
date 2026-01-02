#include "motion_features.h"
#include <math.h>
#include <float.h>

void extract_motion_features(float *ax, float *ay, float *az, uint32_t len, MotionFeatures *out) {
    float sx = 0, sy = 0, sz = 0, sma = 0;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;
    float min_x =  FLT_MAX, min_y =  FLT_MAX, min_z =  FLT_MAX;

    // 1. Pass: Means, SMA, Max, Min
    for (uint32_t i = 0; i < len; i++) {
        sx += ax[i]; sy += ay[i]; sz += az[i];
        sma += fabsf(ax[i]) + fabsf(ay[i]) + fabsf(az[i]);

        if (ax[i] > max_x) max_x = ax[i]; if (ax[i] < min_x) min_x = ax[i];
        if (ay[i] > max_y) max_y = ay[i]; if (ay[i] < min_y) min_y = ay[i];
        if (az[i] > max_z) max_z = az[i]; if (az[i] < min_z) min_z = az[i];
    }

    out->mean_x = sx / len;
    out->mean_y = sy / len;
    out->mean_z = sz / len;
    out->max_x = max_x; out->max_y = max_y; out->max_z = max_z;
    out->min_x = min_x; out->min_y = min_y; out->min_z = min_z;
    out->sma = sma / len;

    // 2. Pass: Standard Deviation
    float vx = 0, vy = 0, vz = 0;
    for (uint32_t i = 0; i < len; i++) {
        vx += powf(ax[i] - out->mean_x, 2);
        vy += powf(ay[i] - out->mean_y, 2);
        vz += powf(az[i] - out->mean_z, 2);
    }
    out->std_x = sqrtf(vx / len);
    out->std_y = sqrtf(vy / len);
    out->std_z = sqrtf(vz / len);
}