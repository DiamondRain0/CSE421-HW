#include <math.h>
#include "kws_inference.h"

// Reproduces Python logic: mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
void normalize_mfccs(float* mfccs, int length) {
    float sum = 0.0, sq_sum = 0.0;
    
    for (int i = 0; i < length; i++) sum += mfccs[i];
    float mean = sum / length;
    
    for (int i = 0; i < length; i++) {
        float diff = mfccs[i] - mean;
        sq_sum += diff * diff;
    }
    float std = sqrtf(sq_sum / length);
    
    if (std < 1e-6f) std = 1.0f; // Prevent division by zero

    for (int i = 0; i < length; i++) {
        mfccs[i] = (mfccs[i] - mean) / std;
    }
}