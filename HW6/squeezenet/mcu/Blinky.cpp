/*------------------------------------------------------------
 * STM32F746 – SqueezeNet FLOAT Inference (CORRECT)
 *-----------------------------------------------------------*/

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "main.h"
#include "cmsis_os2.h"
#include "cmsis_vio.h"
extern int stdin_getchar(void);
extern const unsigned char g_mnist_model_data[];
#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMG_W 32
#define IMG_H 32
#define IMG_C 3
#define IMG_SIZE (IMG_W * IMG_H * IMG_C)

#define TENSOR_ARENA_SIZE (160 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter *interpreter;
static TfLiteTensor *input;
static TfLiteTensor *output;

/*---------------- Inference Thread ----------------*/
static __NO_RETURN void inference_thread(void *arg)
{
    (void)arg;

    printf("Inference online\n");
    printf("Input type=%d\n", input->type);

    for (;;) {
        printf("[READY]\n");

        // Read exactly 32x32x3 bytes
        for (int i = 0; i < IMG_SIZE; i++) {
            uint8_t pixel = (uint8_t)stdin_getchar();

            // DO NOT invert
            // DO NOT subtract mean
            // Match Keras exactly
            input->data.f[i] = (float)pixel * (1.0f / 255.0f);
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            continue;
        }

        int best = 0;
        float best_val = output->data.f[0];

        for (int i = 1; i < 10; i++) {
            float v = output->data.f[i];
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }

        printf("PREDICTED_CLASS:%d\n", best);
    }
}

/*---------------- Main ----------------*/
int app_main(void)
{
    static tflite::MicroMutableOpResolver<8> resolver;

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddConcatenation();
    resolver.AddRelu();
    resolver.AddMean();       // GlobalAveragePooling
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddPad();

    const tflite::Model *model = tflite::GetModel(g_mnist_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Schema mismatch\n");
        while (1);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("TENSOR ALLOCATION FAILED\n");
        while (1);
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    osKernelInitialize();
    osThreadNew(inference_thread, NULL, NULL);
    osKernelStart();
    for (;;);
}
