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

#define IMG_W 96
#define IMG_H 96
#define IMG_C 1
#define IMG_SIZE (IMG_W * IMG_H * IMG_C)

#define NUM_CLASSES 5

// EfficientNet requires large arena
#define TENSOR_ARENA_SIZE (200 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;

// ==============================
// Inference Thread
// ==============================
static __NO_RETURN void inference_thread(void* arg)
{
    (void)arg;

    printf("Rice EfficientNet INT8 online\n");

    for (;;)
    {
        printf("[READY]\n");

        uint8_t* in = input->data.uint8;

        // Read IMG_SIZE bytes from stdin
        for (int i = 0; i < IMG_SIZE; i++) {
            int c = stdin_getchar();
            if (c < 0) { i--; continue; }

            // Normalize and quantize properly
            float f = ((float)c) / 255.0f; // 0..1
            in[i] = (uint8_t)(f / input->params.scale + input->params.zero_point);
        }

        // Invoke the model
        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed!\n");
            continue;
        }

        // Output is uint8 [0..255]
        uint8_t* out = output->data.uint8;

        int best = 0;
        uint8_t best_val = 0;

        for (int i = 0; i < NUM_CLASSES; i++) {
            if (out[i] > best_val) {
                best_val = out[i];
                best = i;
            }
        }

        printf("RESULT:%d\n", best);
    }
}

// ==============================
// Main App
// ==============================
int app_main(void)
{
    static tflite::MicroMutableOpResolver<16> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddLogistic();
    resolver.AddPack();
    resolver.AddStridedSlice();
    resolver.AddShape();
    resolver.AddQuantize();

    const tflite::Model* model = tflite::GetModel(g_efficientnet_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch\n");
        while (1);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed\n");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    osKernelInitialize();
    osThreadNew(inference_thread, NULL, NULL);
    osKernelStart();

    for (;;);
}