#include "kws_inference.h"
#include "kws_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h" 
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <limits>
#include <algorithm>

namespace tflite {
bool HaveSameShapes(const TfLiteTensor* t1, const TfLiteTensor* t2) {
    if (t1->dims->size != t2->dims->size) return false;
    for (int i = 0; i < t1->dims->size; ++i) {
        if (t1->dims->data[i] != t2->dims->data[i]) return false;
    }
    return true;
}

RuntimeShape GetTensorShape(const TfLiteTensor* tensor) {
    if (tensor == nullptr || tensor->dims == nullptr) return RuntimeShape();
    return RuntimeShape(tensor->dims->size, tensor->dims->data);
}

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context, 
                                               TfLiteFusedActivation activation, 
                                               TfLiteTensor* output, 
                                               int* act_min, int* act_max) {
    const float zp = (float)output->params.zero_point;
    const float scale = output->params.scale;
    int32_t q0 = (int32_t)(zp + roundf(0.0f / (scale ? scale : 1.0f)));
    int32_t q6 = (int32_t)(zp + roundf(6.0f / (scale ? scale : 1.0f)));
    *act_min = -128; *act_max = 127;
    if (activation == kTfLiteActRelu) { if (q0 > *act_min) *act_min = q0; }
    else if (activation == kTfLiteActRelu6) { if (q0 > *act_min) *act_min = q0; if (q6 < *act_max) *act_max = q6; }
    return kTfLiteOk;
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output, const TfLiteFusedActivation& activation,
    int* multiplier, int* shift, int* output_activation_min,
    int* output_activation_max, int* per_channel_multiplier, int* per_channel_shift,
    int output_channels) {
    const double input_product_scale = (double)input->params.scale * (double)filter->params.scale;
    const double effective_output_scale = input_product_scale / (double)output->params.scale;
    int32_t m; int s;
    QuantizeMultiplier(effective_output_scale, &m, &s);
    *multiplier = (int)m; *shift = (int)s;
    return CalculateActivationRangeQuantized(context, activation, output, output_activation_min, output_activation_max);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context, const TfLiteTensor* input,
                                              const TfLiteTensor* filter, const TfLiteTensor* bias,
                                              TfLiteTensor* output, double* multiplier) {
    if (output->params.scale == 0) return kTfLiteError;
    *multiplier = ((double)input->params.scale * (double)filter->params.scale) / (double)output->params.scale;
    return kTfLiteOk;
}

const int kMulInput1Tensor = 0;
const int kMulInput2Tensor = 1;
const int kMulOutputTensor = 0;
} // namespace tflite

extern "C" void normalize_mfccs(float* mfccs, int length);

#define TENSOR_ARENA_SIZE (120 * 1024) 
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;
static tflite::MicroMutableOpResolver<25> resolver; 

extern "C" void kws_init(void) {
    printf("LOG: Starting TFLite Init...\r\n");
    const tflite::Model* model = tflite::GetModel(kws_model_data);

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape(); 
    resolver.AddPrelu();
    resolver.AddShape();      
    resolver.AddAdd();        
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddStridedSlice();
    resolver.AddMul();
    resolver.AddPack();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("LOG: Error - AllocateTensors Failed!\r\n");
        return;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    printf("LOG: KWS Init Successful\r\n");
}

extern "C" int kws_predict(float* raw_mfccs) {
    if (!interpreter || !input_tensor) return -10;
        
    // 1. Z-Score Normalization
    // If this function is empty, ensure it matches your Python StandardScaler
    normalize_mfccs(raw_mfccs, KWS_FEATURE_LEN);

    // 2. Manual Quantization
    float scale = input_tensor->params.scale;
    int zp = input_tensor->params.zero_point;
    int8_t* input_ptr = input_tensor->data.int8;

    for (int i = 0; i < KWS_FEATURE_LEN; i++) {
        float q_val = (raw_mfccs[i] / (scale ? scale : 1.0f)) + (float)zp;
        if (q_val < -128.0f) q_val = -128.0f;
        if (q_val > 127.0f) q_val = 127.0f;
        input_ptr[i] = (int8_t)q_val;
    }

    // 3. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) return -2;

    // 4. Result Analysis
    int8_t* scores = output_tensor->data.int8;
    int best_class = 0;
    int8_t max_score = -128;
    
    // Optional Debug: See the actual raw scores for classes 0, 1, 2
    // printf("LOG: Raw Scores: [%d, %d, %d]\r\n", scores[0], scores[1], scores[2]);

    for (int i = 0; i < KWS_NUM_CLASSES; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
            best_class = i;
        }
    }
    return best_class;
}