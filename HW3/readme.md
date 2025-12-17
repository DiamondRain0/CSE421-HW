# Part 1: Estimating Future Temperature Values

*(Section 7.6 Application)*

## 1. Introduction

This application implements an embedded **time-series temperature prediction system** using a lightweight linear regression model deployed on a microcontroller (MCU). The objective is to estimate a future temperature value based on a small window of recent historical measurements while operating under strict computational and memory constraints.

Unlike classification tasks, this application produces a **continuous-valued output**, making it representative of real-world sensing and forecasting scenarios such as environmental monitoring. The design emphasizes simplicity, determinism, and efficient deployment on resource-constrained hardware.

---

## 2. Embedded Processing Pipeline

The temperature regression application follows a streamlined three-stage pipeline:

1. **Data Acquisition and Transport**
2. **Feature Preparation and Normalization**
3. **Regression Inference**

All stages are designed to minimize overhead while preserving predictive accuracy.

---

### 2.1 Data Acquisition and Transport

Temperature data is transmitted from a host PC to the MCU via **UART communication at 115200 baud**. For each inference cycle, the host sends **five lagged temperature readings**, corresponding to time steps ( t-5 ) through ( t-1 ).

Each value is received sequentially and stored in a fixed-length buffer:

```c
float input_features_buffer[5];
```

This design results in an input size of **20 bytes per prediction**, ensuring low communication latency and efficient memory usage.

---

### 2.2 Feature Representation and Normalization

No explicit feature extraction is required. The lagged temperature values themselves form the input feature vector:

[
X = [x_{t-5}, x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}]
]

To ensure consistency with the training data distribution, **Z-score normalization** is applied on the MCU when required. The normalization parameters (mean and standard deviation) are hardcoded using values obtained during offline training.

---

### 2.3 Regression Inference

Prediction is performed using a **linear regression model**, implemented as a single C function:

[
y = W \cdot X + B
]

where (W) and (B) are fixed coefficients stored in flash memory. This approach enables fast, deterministic inference without reliance on external machine learning libraries.

---

## 3. Results and Performance Evaluation

The regression model was evaluated using **100 test samples**, comparing predicted temperature values against ground-truth measurements. Offline evaluation was conducted using **pandas and matplotlib**.

<img width="563" height="453" alt="image" src="https://github.com/user-attachments/assets/ffa6b154-194b-48fc-b73e-7ec07c3ab2df" />

### 3.1 Prediction Accuracy

The model achieves a **mean absolute error (MAE) of 0.0947 °C**, with a maximum observed error of **0.4002 °C**. The predicted temperature distribution closely matches the actual distribution, with nearly identical means and standard deviations.


---

### 3.2 Error Distribution and Bias

Percentile-based analysis shows:

* 50% of predictions have an absolute error below **0.060 °C**
* 75% of predictions have an error below **0.119 °C**
* 95% of predictions fall below **0.328 °C**

The mean prediction bias is **−0.0274 °C**, indicating a slight underestimation tendency that is negligible in practical terms.

---

### 3.3 Trend Capture

The correlation coefficient between actual and predicted temperatures is **0.9987**, confirming that the regression model accurately captures temporal trends in the data despite its simplicity.

---

## 4. Conclusion

This application demonstrates that **accurate time-series temperature prediction** can be achieved on a resource-constrained MCU using a simple lag-based linear regression model. With minimal input features, deterministic execution, and low memory usage, the system provides reliable continuous-value predictions suitable for embedded sensing applications.

---

# Part 2: Human Activity Recognition (HAR) Using Time-Domain Features

*(Section 10.7 Application)*

## 1. Introduction

This application implements an embedded **Human Activity Recognition (HAR)** system using a lightweight single-neuron classifier deployed on a microcontroller (MCU). The objective is to distinguish between two activity classes, **walking** and **non-walking**, based on time-domain features extracted from tri-axial accelerometer data, while operating under strict computational and memory constraints.

The application demonstrates that classical statistical features, when combined with a simple logistic classifier, can provide meaningful activity recognition performance without the need for deep neural networks or computationally expensive feature representations. The design prioritizes deterministic execution, low latency, and minimal memory usage.

---

## 2. Embedded Processing Pipeline

The HAR system follows a three-stage processing pipeline optimized for embedded execution:

1. **Data Acquisition and Transport**
2. **Feature Extraction and Standardization**
3. **Sigmoid-Based Classification**

Each stage is implemented using fixed-size buffers and static memory allocation to ensure predictable runtime behavior on the MCU.

---

### 2.1 Data Acquisition and Transport

Raw accelerometer data is transmitted from the host PC to the MCU via **UART communication at 115200 baud**. Each inference cycle consists of **80 samples per axis (X, Y, Z)**, resulting in a total of **240 floating-point values** per window.

The data is received in a single batch and stored in dedicated buffers:

```c
float acc_x[80];
float acc_y[80];
float acc_z[80];
```

This fixed-window approach ensures consistent feature computation across all inference cycles and simplifies buffer management on the MCU.

---

### 2.2 Feature Extraction and Standardization

Feature extraction is performed entirely on the MCU using the `motion_features.c` module. For each accelerometer window, the following **time-domain features** are computed:

* Mean acceleration (X, Y, Z)
* Standard deviation (X, Y, Z)
* Positive sample count (X, Y, Z)
* Signal Magnitude Area (SMA)

These operations produce a **10-dimensional feature vector** that captures both signal intensity and variability associated with human motion.

To ensure compatibility with the training distribution, **Z-score normalization** is applied using hardcoded mean (μ) and standard deviation (σ) values derived from the offline training dataset. This step prevents feature scale imbalance and improves classifier stability.

---

### 2.3 Classification Inference

Classification is performed using a **single-neuron logistic regression model**, implemented as a compact C function:

[
P = \sigma(W \cdot X + B)
]

where (P) represents the probability that the current window corresponds to the walking activity. A fixed decision threshold of **0.5** is applied to generate the final binary output.

All model parameters are stored in flash memory, enabling fast and deterministic inference without reliance on external machine learning libraries.

---

## 3. Results and Performance Evaluation

The HAR classifier was evaluated on **500 accelerometer windows**, with predictions logged and analyzed offline using **pandas and matplotlib**.

<img width="571" height="453" alt="image" src="https://github.com/user-attachments/assets/0ba168b3-684b-44cd-a6a1-db9a1ffd4d18" />



---

### 3.1 Overall Classification Accuracy

The model achieved a **final classification accuracy of 75.40%**, correctly classifying **377 out of 500 samples**. The running accuracy converges after the initial evaluation window and remains stable between **74% and 76%**, indicating consistent model behavior over time.

---

### 3.2 Confusion Matrix Analysis

The confusion matrix for the evaluation dataset is summarized as follows:

* **True Positives (TP):** 112
* **True Negatives (TN):** 265
* **False Positives (FP):** 56
* **False Negatives (FN):** 67

The classifier exhibits stronger performance on the non-walking class, with slightly more false negatives than false positives. This reflects a conservative decision boundary, which is typical for linear classifiers operating with a fixed threshold.

---

### 3.3 Stability and Model Behavior

The mean running accuracy across all samples is **74.69%**, with a standard deviation of **3.30%**, indicating limited performance variation throughout the evaluation. Additionally, the predicted class distribution closely follows the ground-truth distribution, suggesting minimal class bias.

---

## 4. Conclusion

This application demonstrates that **time-domain accelerometer features combined with a single-neuron classifier** can provide reliable human activity recognition on a resource-constrained MCU. Despite its simplicity, the system achieves stable accuracy, deterministic execution, and low computational overhead, making it well-suited for real-time embedded HAR applications.


---

# Part 3: MFCC Keyword Spotting (Binary Audio Classification)

*(Section 10.8 Application)*

## 1. Introduction

This application implements a **binary keyword spotting (KWS) system** deployed on a microcontroller (MCU), designed to detect a specific target keyword (**Class 1**) against all other audio inputs (**Class 0**). The system relies on spectral-domain features and represents one of the most computationally demanding applications in the project due to the required **Digital Signal Processing (DSP)** operations.

The goal of this application is to demonstrate that robust audio-based classification can be achieved entirely on an embedded platform, without reliance on external processing, while maintaining high reliability and low false activation rates.

---

## 2. Embedded Processing Pipeline

The KWS pipeline transforms **16 KB of raw audio data** into a compact feature representation suitable for classification. All processing stages are executed on the MCU using the `mfcc.c` module and the **CMSIS-DSP** library for optimized FFT and DCT operations.

---

### 2.1 Feature Extraction and Standardization

The MCU performs the complete MFCC pipeline, consisting of:

* Framing and windowing of the raw audio signal
* Spectral analysis using FFT
* Mel filterbank energy computation
* Logarithmic compression
* Discrete Cosine Transform (DCT)

This process produces a **13-dimensional averaged MFCC feature vector**, capturing the spectral characteristics of the 1-second audio input.

To ensure consistency with the offline-trained model, **Z-score normalization** is applied to the feature vector using hardcoded mean (μ) and standard deviation (σ) values stored in flash memory. This step is critical to prevent distribution mismatch and maintain stable classifier behavior on the MCU.

---

### 2.2 Classification Inference

Classification is performed using a **single-neuron logistic regression model**, defined as:

[
P = \sigma(W \cdot X + B)
]

where ( X ) is the normalized MFCC feature vector, and ( W ) and ( B ) are fixed parameters obtained during offline training.

The sigmoid output is thresholded at **0.5** to produce the final binary decision. This design ensures deterministic execution, minimal memory usage, and fast inference suitable for embedded deployment.

---

## 3. Results and Performance Evaluation

The MFCC keyword spotting system was evaluated using **100 labeled audio samples**, containing both target keyword and non-target inputs.

<img width="563" height="433" alt="98348771-9ef7-4caf-beff-aa124005a3b5" src="https://github.com/user-attachments/assets/d165d21e-1462-4df8-823f-26cd9c4a35f1" />

---

### 3.1 Classification Accuracy

The system achieved the following overall performance:

* **Overall Accuracy: 89.0% (89 / 100 correct predictions)**

This result confirms that the MFCC-based pipeline and single-neuron classifier are effective despite the computational constraints of the MCU.

---

### 3.2 Confusion Matrix Analysis

The classification outcomes are summarized below:

| Metric                       | Value | Description                          |
| ---------------------------- | ----- | ------------------------------------ |
| Total Target Samples (P)     | 42    | TP + FN                              |
| Total Non-Target Samples (N) | 58    | TN + FP                              |
| True Positives (TP)          | 32    | Target keyword correctly detected    |
| True Negatives (TN)          | 57    | Non-target audio correctly rejected  |
| False Positives (FP)         | 1     | Non-target audio incorrectly flagged |
| False Negatives (FN)         | 10    | Target keyword missed                |

The confusion matrix highlights a very low false-positive rate, which is a critical requirement for keyword spotting systems.

---

### 3.3 Key Performance Metrics

From the confusion matrix, the following metrics were computed:

* **Precision (Class 1)**
  [
  97.0%
  ]

* **Recall (Class 1)**
  [
   76.2%
  ]

These results indicate that the model strongly prioritizes **precision over recall**. The system is highly reliable at rejecting non-keyword audio, minimizing accidental activations (FP = 1), while accepting a moderate number of missed detections.

---

## 4. Conclusion

The MFCC keyword spotting application successfully integrates a **computationally intensive DSP pipeline** with a **lightweight single-neuron classifier** on an embedded MCU. The system achieves strong overall accuracy and near-perfect precision (97.0%), making it well-suited for **precision-critical, low-power embedded KWS applications**.

The observed trade-off between precision and recall reflects a conservative decision boundary, which is often desirable in real-world keyword spotting scenarios where false activations are more costly than missed detections.

---

# Part 4: Hu Moments Digit Recognition (Image Classification)

*(Section 10.9 Application)*

## 1. Introduction

This application addresses the problem of **Handwritten Digit Recognition** by performing a **binary image classification task**, where the objective is to identify the digit **“0” (Class 1)** against all other digits (**“Not 0”, Class 0**). Rather than relying on raw pixel intensities, the system uses **Hu invariant moments**, which are mathematically defined features invariant to translation, scale, and rotation.

This design choice significantly reduces model complexity and enables accurate image classification using a **single-neuron classifier**, making it well-suited for deployment in **resource-constrained Edge AI environments**. The application validates that classical computer vision descriptors remain effective alternatives to heavier neural architectures when computational and memory budgets are limited.

---

## 2. Embedded Processing Pipeline

The Hu Moments digit recognition pipeline converts raw image pixels into invariant shape descriptors before classification. All stages are executed on the microcontroller (MCU), ensuring full embedded autonomy.

The pipeline consists of:

1. **Image Data Acquisition**
2. **Invariant Feature Extraction (Hu Moments)**
3. **Normalization and Classification Inference**

---

### 2.1 Image Data Acquisition

The host PC transmits a single **28 × 28 grayscale image** (784 pixels) to the MCU. The image is sent as raw, uncompressed **8-bit unsigned integers (`uint8_t`)**, minimizing transmission overhead.

Upon reception, the MCU converts the pixel values to floating-point representation to support moment computation:

```c
uint8_t image_rx_buffer[784];
float image_float_buffer[784]; // Used for moment computation
```

This approach balances low communication cost with sufficient numerical precision during feature extraction.

---

### 2.2 Hu Moment Feature Extraction

Feature extraction is handled by the dedicated `hu.c` module, which performs the following steps:

* **Raw and Central Moment Computation**: Image moments are calculated from pixel intensities. Intermediate computations often use higher precision internally to preserve numerical stability.
* **Hu Invariant Calculation**: Seven (7) Hu moment invariants are derived, providing robustness to translation, rotation, and scaling.
* **Logarithmic Transformation**: Each Hu moment is transformed using
  [
  -\operatorname{sgn}(H_i)\log_{10}(|H_i|)
  ]
  to compress the dynamic range and match the preprocessing performed during offline training.

The result is a **7-dimensional feature vector** that compactly represents the digit’s shape.

Finally, **Z-score normalization** is applied using hardcoded mean (μ) and standard deviation (σ) values obtained from the training dataset.

---

### 2.3 Classification Inference

Inference is performed using a **single-neuron logistic regression model**, defined as:

[
P = \sigma(W \cdot X + B)
]

where ( X ) is the normalized Hu moment feature vector. The sigmoid output is thresholded at **0.5**, producing a binary decision: **Digit “0”** or **Not “0”**.

The small feature dimension (7) enables fast, deterministic inference with minimal memory and computational overhead.

**Deployment Note:** The accuracy of the embedded `log10f` implementation is critical, as small numerical discrepancies introduced during the logarithmic transformation can influence classification outcomes near the decision boundary.

---

## 3. Results and Performance Evaluation

The Hu Moments digit recognition system was evaluated using **100 labeled test images**, containing both digit “0” and non-zero digits.

<img width="563" height="433" alt="bbfb7727-5cbe-4d0a-936c-173f650b8784" src="https://github.com/user-attachments/assets/58fe9168-2ec4-4d9e-82ed-0d6fb34f06e6" />

---

### 3.1 Classification Accuracy

The system achieved the following performance:

* **Overall Accuracy: 85.0% (85 / 100 correct predictions)**

This result confirms that invariant shape-based features can provide meaningful discrimination even with a minimal linear classifier.

---

### 3.2 Confusion Matrix Analysis

The classification outcomes are summarized below:

| Metric                       | Value | Description                         |
| ---------------------------- | ----- | ----------------------------------- |
| Total Target Samples (P)     | 49    | TP + FN                             |
| Total Non-Target Samples (N) | 51    | TN + FP                             |
| True Positives (TP)          | 35    | Digit “0” correctly detected        |
| True Negatives (TN)          | 50    | Non-zero digits correctly rejected  |
| False Positives (FP)         | 1     | Non-zero digit misclassified as “0” |
| False Negatives (FN)         | 14    | Digit “0” missed                    |

The confusion matrix shows a **very low false-positive rate**, indicating strong rejection of non-zero digits.

---

### 3.3 Key Performance Metrics

From the confusion matrix, the following metrics were computed:

* **Precision (Class 1)**
  [
   97.2%
  ]

* **Recall (Class 1)**
  [
  71.4%
  ]

These results indicate that the classifier is **highly precise but conservative**, favoring correct rejection over aggressive detection. This behavior minimizes false positives while accepting some missed detections of the digit “0”.

---

## 4. Conclusion

The Hu Moments Digit Recognition application demonstrates that **classical computer vision techniques** can be effectively deployed on an embedded MCU for image classification. By extracting a compact set of seven invariant features and combining them with careful normalization and a single-neuron classifier, the system achieves **high precision (97.2%)** and **solid overall accuracy (85.0%)**.

The results validate Hu moments as a powerful and efficient alternative to pixel-based or deep-learning approaches in **memory- and power-constrained Edge AI systems**, particularly when false positives are costly and deterministic behavior is required.





