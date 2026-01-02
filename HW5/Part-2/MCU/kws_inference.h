#ifndef KWS_INFERENCE_H
#define KWS_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define KWS_MFCC_COEFFS 40
#define KWS_TIME_FRAMES 32
#define KWS_FEATURE_LEN (KWS_MFCC_COEFFS * KWS_TIME_FRAMES)
#define KWS_NUM_CLASSES 10 

void kws_init(void);
int kws_predict(float *raw_mfccs);

// ADD THIS LINE BELOW:
void extract_mfcc_from_wav(int16_t *audio_data, float *mfcc_out);

#ifdef __cplusplus
}
#endif
#endif