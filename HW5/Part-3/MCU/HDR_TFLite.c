#include "main.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Buffer for 28x28 image (784 bytes)
#define MNIST_IMAGE_SIZE 784
uint8_t image_rx_buffer[MNIST_IMAGE_SIZE];

extern UART_HandleTypeDef huart1;

// C++ Inference function
extern int run_cnn_inference(uint8_t* raw_pixels);

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
    printf("\r\n--- MNIST DIGIT RECOGNITION ONLINE ---\r\n");
    
    // Model initialization is handled inside the C++ wrapper on first call
    printf("LOG: System Initialized\r\n");

    while(1) {
        flush_uart();
        
        // 1. Handshake: Signal Python we are ready
        printf("[READY]\r\n"); 

        // 2. Receive exactly 784 bytes (1 byte per pixel)
        // 10 second timeout
        HAL_StatusTypeDef status = HAL_UART_Receive(&huart1, image_rx_buffer, MNIST_IMAGE_SIZE, 10000);

        if (status == HAL_OK) {
            printf("LOG: Image Received. Running CNN...\r\n");
            
            int pred = run_cnn_inference(image_rx_buffer);
            
            // 3. Send back the prediction
            if (pred >= 0) {
                printf("PREDICTION: %d\r\n", pred);
            } else {
                printf("LOG: Inference Error %d\r\n", pred);
            }
        } 
        else if (status == HAL_TIMEOUT) {
            printf("LOG: UART Timeout\r\n");
        }
        
        HAL_Delay(100); 
    }
    return 0;
}