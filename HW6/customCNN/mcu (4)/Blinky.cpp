#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "main.h"
#include "cmsis_os2.h"
extern int stdin_getchar(void);
extern const unsigned char g_customcnn_model_data[];
#ifdef __cplusplus
}
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define IMG_W 28
#define IMG_H 28
#define IMG_C 1
#define IMG_SIZE (IMG_W*IMG_H*IMG_C)

#define TENSOR_ARENA_SIZE (64*1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter;
static TfLiteTensor* input;
static TfLiteTensor* output;

static __NO_RETURN void inference_thread(void* arg)
{
    (void)arg;

    printf("Tiny CNN INT8 inference online\n");

    for (;;)
    {
        printf("[READY]\n");

        // Fill input
        int32_t in_zero = input->params.zero_point;
        float in_scale = input->params.scale;

        for(int i=0;i<IMG_SIZE;i++){
            int c = stdin_getchar();
            if(c<0){i--; continue;}
            uint8_t pixel = (uint8_t)c;
            input->data.int8[i] = (int8_t)((pixel / 255.0f) / in_scale + in_zero);
        }

        if(interpreter->Invoke()!=kTfLiteOk){
            printf("Invoke failed!\n");
            continue;
        }

        // Read output
        int32_t out_zero = output->params.zero_point;
        float out_scale = output->params.scale;

        float best_val=-1e9f;
        int best=0;
        for(int i=0;i<10;i++){
            float val = (output->data.int8[i]-out_zero)*out_scale;
            if(val>best_val){best_val=val; best=i;}
        }
        printf("PREDICTED_CLASS:%d\n",best);
    }
}

int app_main(void)
{
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMaxPool2D();
    resolver.AddPack();
    resolver.AddShape();
    resolver.AddStridedSlice();

    const tflite::Model* model = tflite::GetModel(g_customcnn_model_data);
    if(model->version()!=TFLITE_SCHEMA_VERSION){
        printf("Schema mismatch\n");
        while(1);
    }

    static tflite::MicroInterpreter static_interpreter(model,resolver,tensor_arena,TENSOR_ARENA_SIZE);
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
