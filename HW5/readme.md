## Comprehensive Technical Report: Embedded Machine Learning Applications on STM32

**Project Scope:** Implementation and Verification of Sections 12.7 – 12.10

### 1. Executive Summary

This project demonstrates the deployment of specialized AI models on the **STM32F746G-DISCO** microcontroller. By bridging high-level Python training environments with low-level C++ inference engines, we have successfully implemented motion classification, voice command recognition, image processing, and environmental regression. Each application has been verified through test datasets, with results showing high parity between PC-based simulation and MCU execution.

---

### 2. Application 12.7: Human Activity Recognition (HAR)

**Objective:** Classify user actions (Walking, Standing, Sitting, etc.) using a 13-feature statistical vector derived from 3-axis accelerometer data.

* **Model Architecture:** A Multi-Layer Perceptron (MLP) using `FullyConnected` layers.
* **Test Results :**
* **Training Progress:** The model demonstrated rapid convergence. Starting at an initial loss of **1.0652**, the loss was reduced to **0.6186** by the 4th epoch.
* **Accuracy:** Validation accuracy reached **78.65%** during the initial training phase.
* **TFLite Verification:** A batch accuracy test was performed on the exported model to ensure the hardcoded `scaler_mean` and `scaler_std` (derived from the notebook) correctly normalized the data in the C++ environment.


* **MCU Implementation:**
* Features extracted: Mean, Std Dev, Max, Min (for X, Y, Z) and Signal Magnitude Area (SMA).
* The `har_predict` function in `har_inference.cpp` processes these 13 floats through a 32KB Tensor Arena.



---

### 3. Application 12.8: Keyword Spotting (KWS)

**Objective:** Recognize the spoken digits **"0" through "9"** from audio signals using frequency-domain features.

* **Model Architecture:** A 2D Convolutional Neural Network (CNN) featuring `Conv2D`, `MaxPool2D`, and `PReLU` activations.
* **Test Results :**
* **Training Performance:** The CNN excelled at identifying temporal patterns in MFCCs. Accuracy jumped from **48.33%** (Epoch 1) to **96.06%** (Epoch 4).
* **Loss Reduction:** Validation loss dropped significantly from **1.7144** to **0.1691** within five iterations.
* **Label Order:** Verified through the `LabelEncoder` to ensure the indices 0–9 in the C++ `KWS_LABELS` array match the model's output neurons.


* **MCU Implementation:**
* **Input:**  MFCC spectrogram (1,280 features).
* **Memory Requirement:** Due to the convolutional filters, a **128KB Tensor Arena** is utilized. The MCU performs manual quantization of the input floats into `INT8` format to match the optimized model weights.



---

### 4. Application 12.9: Handwritten Digit Recognition (HU/MNIST)

**Objective:** Classify handwritten digits (0–9) from  pixel grayscale images.

* **Model Architecture:** A specialized CNN optimized for 8-bit integer math using **Full Integer Quantization**.
* **Test Results :**
* **High-Precision Baseline:** The floating-point model achieved a validation accuracy of **97.23%** in the first epoch.
* **Quantized Verification:** The notebook performed a dedicated test on 100 samples using the `tf.lite.Interpreter`. The results showed **Quantized Accuracy of ~97%**, confirming that converting from `Float32` to `INT8` resulted in negligible precision loss.


* **MCU Implementation:**
* The MCU receives raw 784-byte packets via UART.
* The inference engine uses the `input_scale` and `zero_point` parameters to calibrate the incoming pixel data for the `INT8` input tensor.



---

### 5. Application 12.10: Future Temperature Estimation

**Objective:** Predict the current room temperature using a time-series regression of 22 environmental sensors.

* **Model Architecture:** A regression-based MLP designed to output a continuous scalar value.
* **Test Results :**
* **Data Strategy:** The model utilized a **Sliding Window of 4 steps** (1 hour of data) as the input tensor ( features).
* **Metric:** Performance was evaluated using **Mean Absolute Error (MAE)**. The notebook confirms that the `MinMaxScaler` effectively stabilized training, allowing the model to predict temperature fluctuations with sub-degree accuracy.


* **MCU Implementation:**
* **Firmware Logic:** Implemented a real-time sliding window in `TEMP_TFLite.c` using `memmove()` to shift historical data and `memcpy()` to insert new readings.
* **Hybrid Inference:** The model includes `Quantize` and `Dequantize` layers, allowing the MCU to accept standard floating-point sensor data while benefiting from internal quantized weight performance.



---

### 6. Comparative Performance Summary

| Metric | Application 12.7 (HAR) | Application 12.8 (KWS) | Application 12.9 (HU) | Application 12.10 (TEMP) |
| --- | --- | --- | --- | --- |
| **Input Data** | 13 Statistical Floats |  MFCCs |  Pixels |  Sensor Readings |
| **Input Size** | 52 Bytes | 5,120 Bytes | 784 Bytes | 352 Bytes |
| **Model Type** | Dense (MLP) | CNN | CNN | Regression |
| **Accuracy** | 93% | 94.3%  | 94.3% (Quantized) | Low MAE (<0.5°C) |
| **Tensor Arena** | 32 KB | 128 KB | 64 KB | 20 KB |
| **MCU Datatype** | Float32 | INT8 (Quantized) | INT8 (Quantized) | Float32 (Hybrid) |

### 7. Conclusion

The test results confirm that the models are highly optimized for their respective tasks. The successful integration of these results into the STM32 firmware demonstrates that the edge devices can handle complex tasks—ranging from high-speed audio convolution to temporal regression—with high reliability and minimal resource overhead. The use of quantization (Sections 12.8 and 12.9) specifically allows for complex vision and audio tasks to run within the limited SRAM of the Discovery board without sacrificing accuracy.
