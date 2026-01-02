#ifndef TEMP_INFERENCE_H_
#define TEMP_INFERENCE_H_

#ifdef __cplusplus
extern "C" {
#endif

// This function takes 88 floats (4x22) and returns the predicted temperature
float run_regression_inference(float* input_window);

#ifdef __cplusplus
}
#endif

#endif