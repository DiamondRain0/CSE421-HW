#include "mfcc_neuron_config.h"
#include <math.h>
static const float W[13] = {
    0.00817772f, 0.06540266f, 0.59231174f, 0.93285966f, -0.70721835f, 
    -0.70328397f, -0.31078789f, -0.69182765f, 0.36917180f, 0.40373299f, 
    0.17150873f, -0.04783115f, 0.10335131f, 
};

static const float B = -0.02121617f;

float neuron_predict(float *x) {
    float z = B;
    for(int i=0; i<13; i++) z += x[i] * W[i];
    return 1.0f / (1.0f + expf(-z));
}
