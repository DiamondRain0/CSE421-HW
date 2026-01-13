#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "main.h"
#include "cmsis_os2.h"
extern int stdin_getchar(void);           // keep for data input
extern const unsigned char g_mobilenet_model_data[];
#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// MobileNetV2 input dimensions
#define IMG_W 32
#define IMG_H 32
#define IMG_C 3
#define IMG_SIZE (IMG_W*IMG_H*IMG_C)

// Tensor arena size for MobileNetV2 (~300 KB recommended)
#define TENSOR_ARENA_SIZE (200*1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;

static __NO_RETURN void inference_thread(void* arg)
{
    (void)arg;

    printf("MobileNetV2 INT8 inference online\n");

    for (;;)
    {
        printf("[READY]\n");

        // Fill input via stdin_getchar (32x32x3)
        int32_t in_zero = input->params.zero_point;
        float in_scale = input->params.scale;

        for(int i=0;i<IMG_SIZE;i++){
            int c = stdin_getchar();
            if(c<0){ i--; continue; }      // wait until a byte is available
            uint8_t pixel = (uint8_t)c;
            float normalized = pixel / 255.0f;
            float quantized = normalized / in_scale + in_zero;

            if(quantized > 127.0f) quantized = 127.0f;
            if(quantized < -128.0f) quantized = -128.0f;

            input->data.int8[i] = (int8_t)quantized;
        }

        // Invoke the model
        if(interpreter->Invoke()!=kTfLiteOk){
            printf("Invoke failed!\n");
            continue;
        }

        // Read output
        int32_t out_zero = output->params.zero_point;
        float out_scale = output->params.scale;

        float best_val = -1e9f;
        int best = 0;
        for(int i=0;i<10;i++){
            float val = (output->data.int8[i]-out_zero)*out_scale;
            if(val>best_val){ best_val=val; best=i; }
        }

        printf("PREDICTED_CLASS:%d\n", best);
    }
}

int app_main(void)
{
    // Ops required by MobileNetV2
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddMul();
    resolver.AddMean();      // for GlobalAveragePooling2D
    resolver.AddRelu6();
    resolver.AddLogistic();

    const tflite::Model* model = tflite::GetModel(g_mobilenet_model_data);
    if(model->version()!=TFLITE_SCHEMA_VERSION){
        printf("Schema mismatch\n");
        while(1);
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;

    if(interpreter->AllocateTensors()!=kTfLiteOk){
        printf("Tensor allocation failed\n");
        while(1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    osKernelInitialize();
    osThreadNew(inference_thread,NULL,NULL);
    osKernelStart();

    for(;;);
}
