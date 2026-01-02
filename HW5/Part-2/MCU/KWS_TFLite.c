#include "kws_inference.h"
#include "mfcc.h"
#include "main.h"
#include <stdio.h>
#include <string.h>

#define AUDIO_SAMPLES 16000
#define AUDIO_BYTES 32000 

int16_t audio_rx_buffer[AUDIO_SAMPLES];
float mfcc_features[KWS_FEATURE_LEN]; 
extern UART_HandleTypeDef huart1;

void flush_uart(void) {
    __HAL_UART_CLEAR_OREFLAG(&huart1);
    uint8_t dummy;
    while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE) == SET) {
        dummy = (uint8_t)(huart1.Instance->RDR & 0x00FF);
        (void)dummy;
    }
}

int app_main(void) {
    HAL_Delay(1000); 
    printf("\r\n--- KWS ONLINE ---\r\n");
    kws_init();
    printf("LOG: Model Initialized\r\n");

    while(1) {
        flush_uart();
        printf("[READY]\r\n"); 
        if (HAL_UART_Receive(&huart1, (uint8_t*)audio_rx_buffer, AUDIO_BYTES, 10000) == HAL_OK) {
            printf("LOG: Audio Received. Running Inference...\r\n");
            extract_mfcc_from_wav(audio_rx_buffer, mfcc_features);
            int pred = kws_predict(mfcc_features);
            printf("PREDICTION: %d\r\n", pred+1);
        } 
        HAL_Delay(100); 
    }
    return 0;
}