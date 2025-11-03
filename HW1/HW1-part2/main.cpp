#include "mbed.h"
#include "mfcc.h" // Include your custom MFCC library header

#include <vector>
#include <cmath> // For sin()

// Blinking rate in milliseconds from the original example
#define BLINKING_RATE     500ms
#define M_PI 3.14

int main()
{
    // Initialize the digital pin LED1 as an output for the blinky part
    DigitalOut led(LED1);

    // Use printf for serial output.
    printf("\n--- Mbed OS MFCC with Dummy Value Test ---\n");

    // --- MFCC Parameters ---
    const int FRAME_LENGTH = 1024;
    const int NUM_MEL_FILTERS = 20;
    const int NUM_MFCC_COEFFS = 13;


    // =================================================================
    // STEP 1: MAKE A DUMMY AUDIO VALUE
    // =================================================================
    printf("1. Creating a dummy audio frame of size %d...\n", FRAME_LENGTH);
    
    // We create a std::vector to hold our 16-bit integer audio samples.
    std::vector<int16_t> dummy_audio_data(FRAME_LENGTH);

    // Let's create a sound that's a mix of two frequencies:
    // - A low-frequency hum at 250 Hz
    // - A higher-frequency tone at 1200 Hz
    // This will give us a more interesting result than a single tone.
    float freq1 = 250.0f;
    float freq2 = 1200.0f;
    for (int i = 0; i < FRAME_LENGTH; ++i) {
        // Calculate the value of each sine wave at this point in time
        float sample1 = 0.6f * sin(2.0f * M_PI * freq1 * static_cast<float>(i) / Mfcc::SAMPLE_RATE); // Main tone
        float sample2 = 0.4f * sin(2.0f * M_PI * freq2 * static_cast<float>(i) / Mfcc::SAMPLE_RATE); // Overtone
        
        // Add the waves together and scale the result [-1.0, 1.0] to the int16_t range [-32767, 32767]
        dummy_audio_data[i] = static_cast<int16_t>((sample1 + sample2) * 32767.0f);
    }
    printf("   (Dummy data created with 250Hz and 1200Hz tones)\n");


    // --- Initialize the MFCC processor ---
    // This is done after creating the data, but could be done at any time before computing.
    printf("2. Initializing the MFCC processor...\n");
    Mfcc mfcc_processor(FRAME_LENGTH, NUM_MEL_FILTERS, NUM_MFCC_COEFFS);


    // =================================================================
    // STEP 2: SEND THE DUMMY VALUE TO THE MFCC PROCESSOR
    // =================================================================
    printf("3. Sending dummy data to the MFCC compute method...\n");
    
    // This is the core action: we pass our vector of dummy data to the processor.
    // It returns a new vector containing the floating-point feature values.
    std::vector<float> features = mfcc_processor.compute(dummy_audio_data);


    // --- Print the results ---
    printf("4. MFCC processing complete. Results:\n");
    printf("======================================\n");
    printf("      FINAL MFCC 'FINGERPRINT'\n");
    printf("======================================\n");
    for (size_t i = 0; i < features.size(); ++i) {
        printf("  Coeff %2d:  %f\n", (int)i, features[i]);
    }
    printf("======================================\n");
    printf("Test complete. Now blinking the LED.\n");


    // The original blinky functionality starts here
    while (true) {
        led = !led;
        ThisThread::sleep_for(BLINKING_RATE);
    }
}