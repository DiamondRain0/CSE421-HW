/*------------------------------------------------------------
 * STM32F746 – EfficientNet INT8 Inference (MNIST 28x28x1)
 *-----------------------------------------------------------*/

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "main.h"
#include "cmsis_os2.h"
extern int stdin_getchar(void);
extern const unsigned char g_efficientnet_model_data[];
#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMG_W 28
#define IMG_H 28
#define IMG_C 1
#define IMG_SIZE (IMG_W * IMG_H * IMG_C)

// Tensor arena – increase if you get TENSOR ALLOCATION FAILED
#define TENSOR_ARENA_SIZE (180 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;

/*---------------- Inference Thread ----------------*/
static __NO_RETURN void inference_thread(void* arg)
{
    (void)arg;

    printf("EfficientNet INT8 inference online\n");
    printf("Input type = %d (expect %d)\n", input->type, kTfLiteInt8);

    for (;;) {
        printf("[READY]\n");

        // Fill input tensor
        float in_scale = input->params.scale;
        int32_t in_zero = input->params.zero_point;

        for (int i = 0; i < IMG_SIZE; i++) {
            int c = stdin_getchar();
            if (c < 0) { i--; continue; }
            uint8_t pixel = (uint8_t)c;

            // Convert 0-255 grayscale -> quantized INT8 [-128,127]
            input->data.int8[i] = (int8_t)((((float)pixel / 255.0f) - 0.0f) / in_scale + in_zero);
        }

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed!\n");
            continue;
        }

        // Decode output
        float out_scale = output->params.scale;
        int32_t out_zero = output->params.zero_point;

        float best_val = -1e9;
        int best_class = 0;
        for (int i = 0; i < 10; i++) {
            float val = (output->data.int8[i] - out_zero) * out_scale;
            if (val > best_val) { best_val = val; best_class = i; }
        }

        printf("PREDICTED_CLASS: %d\n", best_class);
    }
}

/*---------------- Main ----------------*/
int app_main(void)
{
    // Ops resolver – only the ops used in your model (smaller footprint)
    static tflite::MicroMutableOpResolver<16> resolver;

    resolver.AddConv2D();           // standard conv
    resolver.AddDepthwiseConv2D();  // depthwise conv
    resolver.AddFullyConnected();   // FC layers
    resolver.AddSoftmax();          // output softmax
    resolver.AddReshape();          // reshape for SE blocks
    resolver.AddPack();             // pack for SE
    resolver.AddShape();            // shape operations
    resolver.AddStridedSlice();     // slicing for SE
    resolver.AddMean();             // global average pooling
    resolver.AddLogistic();         // sigmoid for SE
    resolver.AddMul();              // element-wise multiply for SE
    resolver.AddAdd();              // residual connections

    const tflite::Model* model = tflite::GetModel(g_efficientnet_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch\n");
        while (1);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed\n");
        while (1);
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    osKernelInitialize();
    osThreadNew(inference_thread, NULL, NULL);
    osKernelStart();

    for (;;);
}
