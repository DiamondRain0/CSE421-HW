#include "neuron_config.h"
#include <math.h>

static const float W[10] = {
    0.05246696f, 0.22255537f, 0.17572066f, 0.00740883f, 0.07507453f, 
    -0.04230938f, 0.31040224f, -0.44759721f, 0.79526520f, -0.44294822f, 
    
};

static const float B = -1.26500297f;

float neuron_predict(float *x) {
    float z = B;
    for(int i=0; i<10; i++) z += x[i] * W[i];
    return 1.0f / (1.0f + expf(-z));
}
