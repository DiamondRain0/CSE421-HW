#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "digit_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include <stdio.h>
#include <cmath>
#include <algorithm>

// ====================================================================
// LINKER SHIM BLOCK (Required for Quantized STM32 Projects)
// ====================================================================
namespace tflite {

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
    auto quant = [zp, scale](float f) { return (int)(zp + roundf(f / (scale ? scale : 1.0f))); };
    *act_min = -128; *act_max = 127;
    if (activation == kTfLiteActRelu) { *act_min = std::max(-128, quant(0.0f)); }
    else if (activation == kTfLiteActRelu6) { *act_min = std::max(-128, quant(0.0f)); *act_max = std::min(127, quant(6.0f)); }
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
} 

// ====================================================================
// MNIST INFERENCE LOGIC
// ====================================================================

// 64KB Arena is healthy for a typical MNIST CNN
alignas(16) static uint8_t tensor_arena[64 * 1024]; 
static tflite::MicroInterpreter* interpreter = nullptr;

extern "C" int run_cnn_inference(uint8_t* raw_pixels) {
    static tflite::MicroMutableOpResolver<10> resolver;
    
    if (interpreter == nullptr) {
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddFullyConnected();
        resolver.AddSoftmax();
        resolver.AddQuantize();
        resolver.AddDequantize();
        resolver.AddRelu();

        static tflite::MicroInterpreter static_interpreter(
            tflite::GetModel(digit_model_data), resolver, tensor_arena, sizeof(tensor_arena));
        interpreter = &static_interpreter;
        
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            return -10; // Init failure code
        }
    }

    TfLiteTensor* input = interpreter->input(0);
    
    // Convert incoming 0-255 pixels to model input format
    float scale = input->params.scale;
    int zp = input->params.zero_point;
    for (int i = 0; i < 784; i++) {
        // Pixel (0-1.0) -> Quantize to Int8
        float normalized = (float)raw_pixels[i] / 255.0f;
        int32_t q_val = (int32_t)((normalized / (scale ? scale : 1.0f)) + zp);
        
        // Manual clip to int8 range
        if (q_val > 127) q_val = 127;
        if (q_val < -128) q_val = -128;
        input->data.int8[i] = (int8_t)q_val;
    }

    if (interpreter->Invoke() != kTfLiteOk) return -1;

    TfLiteTensor* output = interpreter->output(0);
    int8_t max_val = -128;
    int digit = 0;
    for (int i = 0; i < 10; i++) {
        if (output->data.int8[i] > max_val) {
            max_val = output->data.int8[i];
            digit = i;
        }
    }
    return digit;
}