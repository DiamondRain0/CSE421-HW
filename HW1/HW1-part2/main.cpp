#include "mbed.h"
#include "mfcc.h"
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CMD_SEND_AUDIO 0x01
#define MAX_SAMPLES 16000  // limit to avoid memory crash (adjust per your RAM)
#define M_PI 3.14159265358979323846

DigitalOut led(LED1);
BufferedSerial pc(USBTX, USBRX, 115200);

bool read_bytes(void* dst, size_t len) {
    uint8_t* ptr = static_cast<uint8_t*>(dst);
    size_t received = 0;
    while (received < len) {
        ssize_t n = pc.read(ptr + received, len - received);
        if (n > 0) {
            received += n;
            led = !led;  // blink on activity
        } else {
            ThisThread::sleep_for(1ms);
        }
    }
    return true;
}

int main() {
    printf("\n--- Mbed MFCC Audio Receiver (Safe) ---\n");

    const int FRAME_LENGTH = 1024;
    const int NUM_MEL_FILTERS = 20;
    const int NUM_MFCC_COEFFS = 13;
    Mfcc mfcc_processor(FRAME_LENGTH, NUM_MEL_FILTERS, NUM_MFCC_COEFFS);

    while (true) {
        uint8_t cmd = 0;
        ssize_t n = pc.read(&cmd, 1);
        if (n <= 0) {
            ThisThread::sleep_for(5ms);
            continue;
        }

        if (cmd == CMD_SEND_AUDIO) {
            printf("[MBED] CMD: SEND_AUDIO received.\n");

            // --- Header ---
            uint32_t num_samples = 0;
            uint32_t sample_rate = 0;
            read_bytes(&num_samples, sizeof(num_samples));
            read_bytes(&sample_rate, sizeof(sample_rate));

            printf("[MBED] Header: %lu samples @ %lu Hz\n",
                   (unsigned long)num_samples, (unsigned long)sample_rate);

            // Validate header
            if (num_samples == 0 || num_samples > MAX_SAMPLES || sample_rate > 96000) {
                printf("[MBED] ⚠️ Invalid header! num_samples=%lu, sample_rate=%lu\n",
                       (unsigned long)num_samples, (unsigned long)sample_rate);
                continue;
            }

            std::vector<int16_t> samples(num_samples);

            // --- Receive Samples ---
            size_t bytes_to_receive = num_samples * sizeof(int16_t);
            size_t received = 0;
            while (received < bytes_to_receive) {
                uint8_t buf[256];
                ssize_t chunk = pc.read(buf, sizeof(buf));
                if (chunk > 0) {
                    memcpy(reinterpret_cast<uint8_t*>(&samples[0]) + received, buf, chunk);
                    received += chunk;
                    led = !led;  // blink per chunk
                } else {
                    ThisThread::sleep_for(1ms);
                }
            }

            led = 1; // steady ON when done
            printf("[MBED] Received %lu bytes.\n", (unsigned long)received);

            // --- Compute MFCC ---
            std::vector<float> features = mfcc_processor.compute(samples);

            printf("======================================\n");
            printf("      FINAL MFCC 'FINGERPRINT'\n");
            printf("======================================\n");
            for (size_t i = 0; i < features.size(); ++i) {
                printf("Coeff %2d:  %f\n", (int)i, features[i]);
            }
            printf("======================================\n");
            printf("[END_OF_FEATURES]\n");
            printf("[MBED] Ready for next file.\n");

            led = 0; // turn off
        }
    }
}
