#ifndef HAR_INFERENCE_H
#define HAR_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

// MUST be 13 to match: Mean(3) + Std(3) + Max(3) + Min(3) + SMA(1)
#define HAR_FEATURE_LEN 13
#define HAR_NUM_CLASSES 6

void har_init(void);
int har_predict(float *features);

#ifdef __cplusplus
}
#endif

#endif