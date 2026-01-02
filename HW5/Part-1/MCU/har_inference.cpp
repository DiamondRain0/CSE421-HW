#include "har_inference.h"
#include "har_model_data.h"
#include <math.h>

/* TFLM core */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TENSOR_ARENA_SIZE (32 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// We only need 3 ops now. No Quantize/Dequantize needed!
static tflite::MicroMutableOpResolver<3> resolver;

// Ensure these match your Python output exactly
float scaler_mean[13] = {0.51492155958734, 0.6800644083957155, 0.5139762513951088, 0.07422821042691657, 0.08259200361197701, 0.06597407728290483, 0.6646638849324351, 0.8338006332349814, 0.6836338181333433, 0.3699998136573051, 0.5101317889780111, 0.38362613418834807, 1.7089622193781666};
float scaler_std[13] = {0.11348377672350908, 0.09627593933077606, 0.06950648134687744, 0.04647531713452688, 0.042726408946011, 0.03095496039363151, 0.14083906671786833, 0.10192102216221496, 0.0804269672720422, 0.13975024851788742, 0.1377396651492351, 0.11796965116130931, 0.1562311717581697};

extern "C" void har_init(void) {
    const tflite::Model* model = tflite::GetModel(har_model_data);
    
    // Only these 3 are required for a standard Dense Float32 model
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return;

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

extern "C" int har_predict(float* features) {
    if (!interpreter || !input_tensor) return -1;

    // Feed raw scaled floats
    for (int i = 0; i < 13; i++) {
        float scaled = (features[i] - scaler_mean[i]) / scaler_std[i];
        input_tensor->data.f[i] = scaled; 
    }

    if (interpreter->Invoke() != kTfLiteOk) return -1;

    // Output is float softmax
    int best_class = 0;
    float max_score = output_tensor->data.f[0];
    for (int i = 1; i < 6; i++) {
        if (output_tensor->data.f[i] > max_score) {
            max_score = output_tensor->data.f[i];
            best_class = i;
        }
    }
    return best_class;
}