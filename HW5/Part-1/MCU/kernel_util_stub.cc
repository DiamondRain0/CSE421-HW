// Stub implementations for missing TFLite kernel utility functions
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {

// Get tensor shape helper
RuntimeShape GetTensorShape(const TfLiteTensor* tensor) {
    if (!tensor) {
        return RuntimeShape();
    }
    RuntimeShape shape(tensor->dims->size);
    for (int i = 0; i < tensor->dims->size; i++) {
        shape.SetDim(i, tensor->dims->data[i]);
    }
    return shape;
}

// Calculate quantized convolution multiplier
TfLiteStatus GetQuantizedConvolutionMultipler(
    TfLiteContext* context,
    const TfLiteTensor* input,
    const TfLiteTensor* filter,
    const TfLiteTensor* bias,
    TfLiteTensor* output,
    double* multiplier) {
    
    const double input_product_scale = 
        static_cast<double>(input->params.scale) * 
        static_cast<double>(filter->params.scale);
    const double output_scale = static_cast<double>(output->params.scale);
    
    if (output_scale == 0) {
        return kTfLiteError;
    }
    
    *multiplier = input_product_scale / output_scale;
    return kTfLiteOk;
}

// Calculate activation range for quantized operations
TfLiteStatus CalculateActivationRangeQuantized(
    TfLiteContext* context,
    TfLiteFusedActivation activation,
    TfLiteTensor* output,
    int* act_min,
    int* act_max) {
    
    int32_t q_min = 0;
    int32_t q_max = 0;
    
    if (output->type == kTfLiteUInt8) {
        q_min = 0;
        q_max = 255;
    } else if (output->type == kTfLiteInt8) {
        q_min = -128;
        q_max = 127;
    } else if (output->type == kTfLiteInt16) {
        q_min = -32768;
        q_max = 32767;
    } else {
        return kTfLiteError;
    }
    
    switch (activation) {
        case kTfLiteActNone:
            *act_min = q_min;
            *act_max = q_max;
            break;
        case kTfLiteActRelu:
            *act_min = 0;
            *act_max = q_max;
            break;
        case kTfLiteActReluN1To1:
            // For range [-1, 1]
            *act_min = q_min;
            *act_max = q_max;
            break;
        case kTfLiteActRelu6:
            // For range [0, 6], need to quantize
            *act_min = 0;
            *act_max = q_max; // Simplified
            break;
        case kTfLiteActTanh:
        case kTfLiteActSignBit:
        case kTfLiteActSigmoid:
        default:
            *act_min = q_min;
            *act_max = q_max;
            break;
    }
    
    return kTfLiteOk;
}

}  // namespace tflite