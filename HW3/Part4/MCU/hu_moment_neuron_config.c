#include "hu_moment_neuron_config.h"
#include <math.h>

static const float W[7] = {
    -1.64979887f, 2.05105972f, 11.68839836f, 2.17387938f, -0.08829932f, 
    0.13784400f, 0.21821508f, 
};

static const float B = -4.16978741f;

float neuron_predict(float *x) {
    float z = B;
    for(int i=0; i<7; i++) z += x[i] * W[i];
    return 1.0f / (1.0f + expf(-z));
}
