# **Embedded Machine Learning Classifiers on STM32F7**
## 1. Introduction
Machine learning on microcontrollers has become increasingly feasible due to significant advancements in embedded processing power, hardware floating-point support, and optimized Digital Signal Processing (DSP) libraries. However, deploying classical machine learning models—specifically Bayes classifiers, k-Nearest Neighbors (k-NN), and Support Vector Machines (SVMs)—on resource-constrained devices presents distinct engineering challenges. Strict limitations on SRAM and Flash memory, the absence of an operating system (bare-metal execution), and the requirement for deterministic timing necessitate a meticulous approach to both software architecture and algorithmic efficiency.

This project presents the design and implementation of three embedded machine-learning applications—Human Activity Recognition, Audio Keyword Spotting, and Handwritten Digit Recognition—on the STM32F746G-DISCO development board. The methodology follows an "offline training, on-device inference" paradigm: datasets are preprocessed and models are trained in a high-level Python environment before being transpiled into optimized C code for deployment. To validate the system, a Hardware-in-the-Loop (HIL) architecture was established, allowing a host PC to stream feature vectors or raw data segments to the microcontroller. This approach ensures reproducible testing against standard datasets while respecting the physical constraints of the embedded target.

The primary objectives of this work are to establish a unified firmware framework capable of executing multiple distinct classifiers, to leverage hardware-acceleration via the STM32F7’s Floating Point Unit (FPU) and ARM CMSIS-DSP library, and to ensure system correctness by validating embedded inference results against reference Python models.

## 2. Development Environment
The firmware was developed using **Visual Studio Code** with the **Keil Studio** extension, providing a modern integrated development environment (IDE) compatible with the ARM Cortex-M7 architecture. The target microcontroller, an STM32F746NG, operates at 216 MHz and features a hardware FPU, which is critical for the efficient execution of the floating-point arithmetic required by the implemented classifiers.

To achieve real-time performance, the project integrates the **ARM CMSIS-DSP library**. This library accelerates computationally intensive operations: linear algebra functions (matrix multiplication and transposition) are utilized for the Quadratic Bayes classifier, while optimized Fast Fourier Transform (FFT) routines are employed for audio feature extraction.

Memory management was a significant challenge, particularly given the large arrays required for MFCC processing buffers and SVM support vectors. To resolve this, the linker script was manually modified to increase the reserved Heap and Stack sizes. Furthermore, all constant model parameters (weights, support vectors, and training samples) were explicitly declared `const` to force their storage in Flash memory, thereby preserving the limited SRAM for runtime variables.

## 3. Communication Architecture
Since the STM32F746G-DISCO board lacks the specific sensors required for the datasets used (such as a microphone or accelerometer), a Hardware-in-the-Loop (HIL) setup was implemented. In this architecture, the PC acts as the sensor data source, streaming raw or preprocessed samples to the MCU via UART, where feature extraction and classification are performed in real-time.

To prevent buffer overflows caused by baud rate mismatches or inference latency, a robust **Stop-and-Wait handshake protocol** was designed. The communication flow is as follows: the MCU transmits a `[READY]` signal to the host; the host responds by transmitting a single chunk of data; upon successful reception, the MCU acknowledges with `[SAMPLE_OK]`, blocks further reception to perform inference, and finally transmits the predicted class label. This protocol enforces synchronization, ensures deterministic execution timing, and facilitates automated accuracy evaluation over large test sets.

<img width="648" height="388" alt="Communication Flowchart" src="https://github.com/user-attachments/assets/d45c1b17-d7b4-42fc-b722-e42d65712683" />



**FIGURE 3.1 — Communication Flowchart**

## 4. End-to-End Machine Learning Pipeline
The three applications share a unified development workflow that bridges high-level Python research tools with low-level embedded C execution.

1.  **Data Preparation:** Raw data is ingested on the PC, cleaned, and formatted. This includes time-series segmentation for accelerometer data, 16-bit PCM conversion for audio, and thresholding/binarization for image data.
2.  **Feature Extraction:** To reduce dimensionality, specific features are extracted: statistical descriptors (Mean, Std Dev, SMA) for Activity Recognition; 13-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) for Audio; and seven Hu Moment Invariants for Digit Recognition.
3.  **Model Conversion:** Classifiers are trained using `scikit-learn` and converted into C-compatible header files using `sklearn2c`. These generated files contain the learned model parameters (means, covariance matrices, support vectors) stored as static arrays.
4.  **Integration and Inference:** The generated C files are integrated into the firmware. Using the HIL interface, the PC streams test samples to the MCU. The MCU computes features on-the-fly and executes the inference logic.
5.  **Validation:** The embedded predictions are returned to the PC and compared against the reference Python model outputs to verify mathematical correctness.

<img width="627" height="270" alt="End-to-End ML Pipeline Diagram" src="https://github.com/user-attachments/assets/aabd2eaf-7e77-4531-9aa6-24effd2f9302" />

**FIGURE 4.1 — End-to-End ML Pipeline Diagram**


## 5. Application 1 — Human Activity Recognition (Bayes Classifier)
The Human Activity Recognition (HAR) application is designed to classify human movement patterns based on tri-axial accelerometer data. This embedded implementation utilizes a **Quadratic Discriminant Analysis (QDA)** Bayes classifier, which models each activity class as a multivariate Gaussian distribution with distinct covariance matrices. The entire inference pipeline, from feature extraction to classification, is executed in real-time on the STM32F746G microcontroller.

### 5.1 Dataset and Training
The model was trained using the **WISDM (Wireless Sensor Data Mining)** dataset, which contains labeled accelerometer recordings for six distinct activities: Downstairs, Jogging, Sitting, Standing, Upstairs, and Walking. The continuous time-series signals were segmented into overlapping windows of **128 samples** with a 50% step size. Feature extraction and model training were conducted in a Python environment using `scikit-learn`. The trained model parameters—specifically the class means, inverse covariance matrices, and log-determinants—were transpiled into C code using `sklearn2c` and stored in the microcontroller's Flash memory.

### 5.2 Feature Extraction
To reduce the high-dimensional raw accelerometer data into a compact and discriminative representation, each 128-sample window is transformed into a 10-dimensional feature vector on the MCU. The extracted features include:
*   **Mean:** Calculated per axis to capture the static component of gravity, indicating device orientation.
*   **Standard Deviation:** Calculated in the time domain to measure the intensity and dynamic range of the motion.
*   **Positive Counts:** Used as a computationally efficient proxy for the signal frequency or zero-crossing rate.
*   **Signal Magnitude Area (SMA):** Aggregates the magnitude of acceleration across all axes to quantify overall physical energy expenditure.

### 5.3 Embedded Classifier Implementation
The QDA classifier computes the posterior probability of each class for the incoming feature vector. The core of this computation involves calculating the **Mahalanobis distance**, which requires heavy matrix algebra. To ensure real-time performance, these operations were implemented using the **ARM CMSIS-DSP library** (specifically `arm_mat_mult_f32` and `arm_mat_trans_f32`), leveraging the STM32F7’s hardware Floating Point Unit (FPU). The MCU receives feature vectors via the established handshake protocol, computes the discriminant scores, and transmits the predicted class label back to the PC.

### 5.4 Performance Analysis
Validation was performed using a withheld subset of the WISDM dataset, yielding an overall system accuracy of **75.6%**. The system demonstrated high efficacy in recognizing dynamic activities, with Jogging and Walking achieving recognition rates of 94.48% and 80.00% respectively. Static activities such as Sitting and Standing were also classified with high accuracy due to distinct orientation signatures. However, activities with biomechanically similar signatures, specifically climbing stairs (Upstairs/Downstairs), exhibited higher misclassification rates.

<img width="740" height="333" alt="Per-Class Accuracy Histogram" src="https://github.com/user-attachments/assets/18ca12b7-6d90-45b4-9862-18d484e9ed0f" />

**FIGURE 5.1 — Per-Class Accuracy Histogram**

<img width="686" height="456" alt="Confusion Matrix Heatmap" src="https://github.com/user-attachments/assets/e6eeb988-e046-473a-a4dd-9fdab8963a14" />

**FIGURE 5.2 — Confusion Matrix Heatmap**

The confusion matrix analysis confirms that while the QDA classifier is robust for distinguishing between static and dynamic states, discriminating between specific dynamic subsets requires features more sensitive to vertical displacement or barometric pressure.



## 6. Application 2 — Audio Keyword Spotting (k-NN)
The second application performs real-time classification of spoken digits (0–9) utilizing the **Free Spoken Digit Dataset (FSDD)**. In contrast to the parametric Bayes model, this application deploys a non-parametric **k-Nearest Neighbors (k-NN)** classifier directly on the embedded target.

### 6.1 Dataset and Configuration
The dataset consists of 8kHz audio recordings trimmed of silence. Audio files were converted to 16-bit PCM format on the host PC. Due to the limited SRAM available on the microcontroller, the full training set could not be stored in RAM. Instead, a representative subset of **500 training samples** was selected and stored as a constant array in the microcontroller's Flash memory.

### 6.2 On-Device MFCC Feature Extraction
To enable accurate audio classification, the microcontroller extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** in real-time. The incoming audio stream is segmented into 1024-sample frames with a hop size of 512 samples. The feature extraction pipeline utilizes **CMSIS-DSP** for the Fast Fourier Transform (FFT) via `arm_rfft_fast_f32`. The power spectrum is mapped onto a 20-band Mel-scale filterbank, followed by a Discrete Cosine Transform (DCT) to yield 13 decorrelated coefficients per frame. These frame-wise vectors are averaged to produce a single 13-dimensional signature for the spoken utterance.

### 6.3 k-NN Classifier
The inference engine implements a k-NN search with **$k=3$**. It calculates the squared Euclidean distance between the live input vector and the 500 stored training vectors. The system identifies the three nearest neighbors and performs a majority vote to determine the predicted digit.

### 6.4 Embedded Implementation
The MCU utilizes a circular buffer to receive streaming audio via the UART handshake protocol. Large temporary buffers required for the FFT and Mel computations are allocated on the system Heap to prevent Stack overflow. This architecture ensures that audio processing and classification occur within the timing constraints of the handshake protocol.

### 6.5 Performance Analysis
System validation was conducted using the `test_knn_audio.py` script, which streamed random WAV files from the test set. The system achieved an overall accuracy of approximately **79.67%**.


<img width="746" height="340" alt="Per-Class Accuracy Histogram" src="https://github.com/user-attachments/assets/b3099b46-fa72-4ca9-8223-659fc0b5add7" />

**FIGURE 6.1 — Per-Class Accuracy Histogram**


Performance varied across classes; for instance, the digit "4" was recognized with **97.50%** accuracy, whereas digit "3" proved more challenging with **57.58%** accuracy. The confusion matrix highlights that errors are concentrated among digits with acoustically similar phonetic structures.


<img width="801" height="515" alt="Confusion Matrix Heatmap" src="https://github.com/user-attachments/assets/75b1a75c-5ddc-410b-8a22-eb384b460c4c" />

**FIGURE 6.2 — Confusion Matrix Heatmap**


<img width="575" height="942" alt="Python Terminal Output Screenshot" src="https://github.com/user-attachments/assets/ba8b2296-2802-4bfc-a737-8b2dafad61d7" />

**FIGURE 6.3 — Python Terminal Output Screenshot**


## 7. Application 3 — Handwritten Digit Recognition (SVM)
The final application implements an image classification system for recognizing handwritten digits (0–9) using the **MNIST dataset**. To make this computationally tractable on a microcontroller, the system combines **Hu Moment Invariants** for dimensionality reduction with a **Support Vector Machine (SVM)** classifier.

### 7.1 Dataset and Preprocessing
The system processes $28\times28$ pixel grayscale images. The raw pixel data is streamed byte-by-byte from the PC to the MCU. To facilitate shape analysis, the MCU performs on-device binarization: pixels exceeding a threshold of 128 are set to 1 (foreground), while others are set to 0 (background).

### 7.2 Feature Extraction: Hu Moments
Processing 784 raw pixels directly is inefficient for a standard SVM on an embedded target. Instead, the firmware calculates **seven Hu Moment Invariants**. These descriptors are mathematically derived from central normalized moments and are invariant to image translation, scaling, and rotation. A signed logarithmic transformation is applied to the moments to compress their dynamic range, resulting in a compact 7-dimensional feature vector that describes the topological shape of the digit.

### 7.3 SVM Classifier
The classifier is a **Kernel SVM** with a Radial Basis Function (RBF), trained in Python using a "one-vs-one" strategy. This results in 45 binary classifiers. The model comprises **1,928 support vectors**, which are stored in the STM32's Flash memory. During inference, the MCU computes the RBF kernel distance between the input feature vector and all support vectors, aggregating votes from the 45 binary classifiers to predict the final digit.

### 7.4 Embedded Implementation
The inference logic is encapsulated in `svm_logic.c`. To ensure numerical stability, input features are standardized using mean and scale parameters derived during training. Kernel distance computations utilize pre-allocated static arrays to prevent stack exhaustion. The prediction result is transmitted back to the PC for validation.

### 7.5 Performance Analysis
Testing with 500 unseen MNIST images yielded an overall accuracy of **49.4%**. While digits with distinct topological features (such as "1") were recognized with **100%** accuracy, structurally complex digits like "2" (8.3%) and "5" (19.5%) suffered significant misclassification.

<img width="709" height="355" alt="Per-Class Accuracy Histogram" src="https://github.com/user-attachments/assets/3e301d86-6e97-47e8-a581-805d141ed412" />

**FIGURE 7.1 — Per-Class Accuracy Histogram**

<img width="728" height="487" alt="Confusion Matrix Heatmap" src="https://github.com/user-attachments/assets/8c877f5b-9e37-4f98-908e-e090eb3a2803" />

**FIGURE 7.1 — Confusion Matrix Heatmap**



This performance indicates that while Hu Moments provide substantial dimensionality reduction (from 784 to 7 dimensions), they may discard fine-grained spatial details required to distinguish between complex handwriting variations in a limited feature space.



## 8. Challenges and Solutions
The development of these embedded machine learning applications presented several recurring engineering challenges, which were addressed as follows:

*   **Memory Overflow:** The allocation of large arrays for MFCC FFT processing and SVM kernel caching initially caused stack collisions and hard faults.
    *   *Solution:* Large buffers were moved to the **Heap** or **BSS** (static allocation), and read-only model weights were explicitly placed in **Flash memory** using the `const` qualifier.
*   **Data Synchronization:** High-speed data streaming from the PC frequently overran the microcontroller's receive buffer during inference operations.
    *   *Solution:* A **Stop-and-Wait UART handshake protocol** (`[READY]` / `[SAMPLE_OK]`) was implemented to enforce flow control and synchronization.
*   **Computational Speed:** Standard C implementations of matrix operations and FFTs were insufficient for real-time performance.
    *   *Solution:* The system was integrated with the **ARM CMSIS-DSP library**, enabling the firmware to leverage the STM32F7's hardware Floating Point Unit (FPU) and DSP instructions.



## 9. Conclusion
This report demonstrates the successful deployment of classical machine-learning classifiers on a resource-constrained STM32F7 microcontroller. By combining robust feature extraction techniques (Statistical moments, MFCCs, and Hu Moments) with hardware-optimized mathematical routines (CMSIS-DSP), the system achieves real-time inference for human activity recognition, audio keyword spotting, and handwritten digit recognition. The implementation of a Hardware-in-the-Loop architecture ensures the reproducibility and validation of embedded predictions against ground-truth models. Overall, this framework provides a practical and scalable foundation for future embedded machine learning applications.
