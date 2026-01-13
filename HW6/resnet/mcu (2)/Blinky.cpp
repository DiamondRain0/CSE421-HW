/*------------------------------------------------------------
 * STM32F746 – ResNet INT8 Inference (MNIST)
 *-----------------------------------------------------------*/

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "main.h"
#include "cmsis_os2.h"
extern int stdin_getchar(void);
extern const unsigned char g_resnet_model_data[];
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

#define TENSOR_ARENA_SIZE (200 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;

/*---------------- Inference Thread ----------------*/
static __NO_RETURN void inference_thread(void* arg)
{
    (void)arg;

    printf("ResNet INT8 inference online\n");
    printf("Input type = %d (expect %d)\n", input->type, kTfLiteInt8);

    const float in_scale = input->params.scale;
    const int   in_zero  = input->params.zero_point;

    for (;;) {
        printf("[READY]\n");

                // Fill input tensor
        float scale = input->params.scale;
        int32_t zero_point = input->params.zero_point;

        for (int i = 0; i < IMG_SIZE; i++) {
            int c = stdin_getchar();
            if (c < 0) { i--; continue; }
            uint8_t pixel = (uint8_t)c;
            input->data.int8[i] = (int8_t)((pixel / 255.0f) / scale + zero_point);
        }

        // Invoke
        interpreter->Invoke();

        // Read output
        float best_val = -1e9;
        int best = 0;
        float out_scale = output->params.scale;
        int out_zero = output->params.zero_point;

        for (int i = 0; i < 10; i++) {
            float val = (output->data.int8[i] - out_zero) * out_scale;
            if (val > best_val) { best_val = val; best = i; }
        }
        printf("PREDICTED_CLASS:%d\n", best);

    }
}

/*---------------- Main ----------------*/
int app_main(void)
{
    static tflite::MicroMutableOpResolver<14> resolver;

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAdd();
    resolver.AddMul();             // batch norm folding
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddShape();           // flatten
    resolver.AddStridedSlice();    // flatten
    resolver.AddPack();            // flatten
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();

    const tflite::Model* model = tflite::GetModel(g_resnet_model_data);
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
