![](./resources/official_armmbed_example_badge.png)

# MFCC Feature Extraction on Mbed OS (Bare Metal)

This example demonstrates how to perform **Mel-Frequency Cepstral Coefficient (MFCC)** feature extraction on embedded devices using **Arm Mbed OS** and **CMSIS-DSP**.  
MFCCs are widely used in **speech recognition**, **keyword spotting**, and **audio classification**.  
This project shows how to compute MFCCs efficiently on a bare-metal embedded system.

You can build this project with all supported [Mbed OS build tools](https://os.mbed.com/docs/mbed-os/latest/tools/index.html).  
The example specifically refers to the command-line interface tool [Arm Mbed CLI](https://github.com/ARMmbed/mbed-cli#installing-mbed-cli).

---

## Application Functionality

The application creates a **dummy audio frame** composed of two sine waves (250 Hz and 1200 Hz) and passes it to the MFCC processor.  
The MFCC algorithm converts this waveform into a **13-element feature vector**, representing the frequency characteristics of the sound.  
After processing, the result is printed via serial, and the LED on the board blinks continuously.

---

## Expected Output

After flashing the binary and opening the serial terminal, you should see something similar to:
```
--- Mbed OS MFCC with Dummy Value Test ---
1. Creating a dummy audio frame of size 1024...
   (Dummy data created with 250Hz and 1200Hz tones)
2. Initializing the MFCC processor...
3. Sending dummy data to the MFCC compute method...
4. MFCC processing complete. Results:
======================================
      FINAL MFCC 'FINGERPRINT'
======================================
  Coeff  0:  -3.210000
  Coeff  1:  12.880000
  Coeff  2:   8.410000
  Coeff  3:   1.750000
  ...
======================================
Test complete. Now blinking the LED.
```

---

## Project Structure

```
├── main.cpp              # Entry point (dummy signal + MFCC test)
├── mfcc.cpp              # MFCC feature extraction implementation
├── mfcc.h                # MFCC class definition and constants
├── mbed-os/              # Mbed OS source
└── README.md             # Project documentation

```
---

## Core Algorithm Steps

1. Windowing – applies a Hamming window to the signal to reduce spectral leakage.

2. FFT – computes the frequency spectrum using CMSIS-DSP’s arm_rfft_fast_f32.

3. Power Spectrum – calculates magnitude squared values from FFT output.

4. Mel Filterbank – applies a triangular filter bank spaced on the Mel scale (human hearing).

5. Logarithm – converts power values to log scale.

6. DCT (Discrete Cosine Transform) – decorrelates features and produces the MFCCs.

7. Output – returns 13 MFCC coefficients as a feature vector.

---

## Configuration

You can adjust these parameters in main.cpp or mfcc.h:

| Parameter         | Description                              | Default |
| ----------------- | ---------------------------------------- | ------- |
| `FRAME_LENGTH`    | FFT length (number of samples per frame) | 1024    |
| `NUM_MEL_FILTERS` | Number of Mel filters                    | 20      |
| `NUM_MFCC_COEFFS` | Number of output MFCC coefficients       | 13      |
| `SAMPLE_RATE`     | Sampling rate (Hz)                       | 8000    |
| `MEL_LOW_FREQ`    | Lowest frequency for Mel filterbank      | 20      |
| `MEL_HIGH_FREQ`   | Highest frequency (Nyquist)              | 4000    |


Example usage: 
```
Mfcc mfcc_processor(FRAME_LENGTH, NUM_MEL_FILTERS, NUM_MFCC_COEFFS);
std::vector<float> features = mfcc_processor.compute(audio_frame);

```
---

## Requirements

- Mbed OS 6.0+

- CMSIS-DSP library (included with Mbed OS)

- C++17 compatible toolchain

- Any Arm Cortex-M board with sufficient RAM (≥ 32 KB recommended)

---

## Troubleshooting

- Missing CMSIS-DSP symbols:
    Ensure CMSIS-DSP is included in mbed_app.json:
    ```
    {
        "target.include_dirs": [
                    "external/CMSIS-DSP/Include"
                ]
    }
    ```

---

## Related Links

[MFCC Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

[CMSIS-DSP Documentation](https://arm-software.github.io/CMSIS_5/DSP/html/index.html)

[Mbed OS Official Docs](https://os.mbed.com/docs/mbed-os/v6.16/introduction/index.html)

[Mbed OS Serial Communication](https://os.mbed.com/docs/mbed-os/v6.16/program-setup/serial-communication.html)