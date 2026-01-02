#include "temp_inference.h"
#include "temp_model_data.h" 
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
    // 20KB is usually plenty for a simple regression model
    const int kTensorArenaSize = 20 * 1024;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

float run_regression_inference(float* input_window) {
    // 1. Initialize the model
    const tflite::Model* model = tflite::GetModel(temp_model_data);
    
    // 2. Register Ops. Regression models use FullyConnected. 
    // If you used specialized layers, add them here.
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();

    // 3. Build the Interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate tensors if not already done
    static bool tensors_allocated = false;
    if (!tensors_allocated) {
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            return -1.0f; 
        }
        tensors_allocated = true;
    }

    // 4. Input: Copy the 88 features into the input tensor
    TfLiteTensor* input = interpreter.input(0);
    for (int i = 0; i < 88; i++) {
        input->data.f[i] = input_window[i];
    }

    // 5. Run Inference
    if (interpreter.Invoke() != kTfLiteOk) {
        return -2.0f;
    }

    // 6. Output: Get the single float prediction
    TfLiteTensor* output = interpreter.output(0);
    return output->data.f[0];
}