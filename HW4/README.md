# Homework 4 Report

## 1. Introduction

This homework focused on transitioning from single-neuron models to Multilayer Perceptrons (MLP) to solve complex classification and regression tasks. We explored four distinct domains:

* Human Activity Recognition(Accelerometer data)
* Keyword Spotting(Audio signals)
* Handwritten Digit Recognition(Image shape descriptors)
* Temperature Estimation(Environment sensors)

---

## 2. Part 1: Human Activity Recognition (HAR)

### 2.1 Methodology

We implemented a classifier for the **WISDM dataset** to recognize activities like walking, jogging, and sitting.

* **Data Cleaning:** The raw dataset presented parsing challenges, including malformed lines and trailing semicolons in the Z-axis data. We utilized a custom `read_data` function with `on_bad_lines='skip'` and a string-replacement routine to ensure a clean numerical pipeline.
* **Feature Extraction:** We segmented the tri-axial accelerometer data into 80-sample windows. Ten statistical features were extracted per window (Mean, Std Dev, and Max for each axis, plus the mean of the axis sum), reducing the input complexity for the MLP.
* **Architecture:** A 3-layer MLP (100-100-6) using ReLU activations and Softmax for classification.

### 2.2 Results

<img width="640" height="480" alt="Part-1" src="https://github.com/user-attachments/assets/8b94c831-9478-483a-a6a3-b219ac70e79d" />

The model successfully converged over 50 epochs. Results were visualized using a Confusion Matrix, demonstrating high precision in distinguishing repetitive movements like "Jogging" from static states like "Sitting." The final model was saved as `mlp_har_model.h5`.

---

## 3. Part 2: Keyword Spotting (KWS)

### 3.1 Methodology

This task focused on recognizing spoken digits (0-9) from the Speech Commands/FSDD recordings.

* **Preprocessing:** Audio signals were normalized to a fixed 1-second duration (8000 samples) to ensure input consistency.
* **Log-Spectral Features:** We transformed the signals into the frequency domain via FFT. To better represent audio, we applied a **Logarithmic Transform** to the frequency bins and utilized `StandardScaler` to normalize the input range. This step was vital to prevent the "random guessing" baseline (10% accuracy) observed in unscaled models.
* **Architecture:** An MLP with 26 input features corresponding to the log-magnitude frequency bins.

### 3.2 Results



By incorporating standardization and log-scaling, the model achieved significant accuracy gains. The trained model was preserved as `mlp_kws_model.h5`.

---

## 4. Part 3: Handwritten Digit Recognition (MNIST via Hu Moments)

### 4.1 Methodology

To simulate an embedded environment with limited memory, we avoided using all 784 pixels of the MNIST digits.

* **Feature Engineering:** We calculated **7 Hu Moments** for each image. These are shape descriptors invariant to rotation, translation, and scale.
* **Normalization:** We applied a log-transform to handle the massive dynamic range of the moments ($10^{-3}$ to $10^{-20}$), followed by standardization.
* **The 6/9 Limitation:** We observed that accuracy is mathematically capped (typically 60-75%) because Hu Moments are rotation-invariant; the model cannot inherently distinguish between a "6" and a "9."

### 4.2 Results

<img width="287" height="35" alt="part-3_result" src="https://github.com/user-attachments/assets/221c8ce7-ce8d-4904-a4e4-618c9f94b12b" />


Despite the 99% reduction in input data (from 784 to 7 features), the model achieved a stable classification performance. The model was saved as `mlp_mnist_model.h5`.

---

## 5. Part 4: Estimating Future Temperature (Regression)

### 5.1 Methodology

Using the **SML2010 dataset**, we built a regressor to predict indoor temperatures.

* **Time-Series Shifting:** The model predicts the current temperature ($t$) based on a window of the 5 previous readings ($t-5$ to $t-1$).
* **Algorithm:** We implemented an MLP Regressor (10-10-1) using the SGD optimizer with a learning rate of 0.005, aiming to minimize Mean Absolute Error (MAE).
* **Normalization:** Input scaling was applied to ensure the different temperature ranges did not bias the gradient updates.

### 5.2 Results


<img width="1200" height="600" alt="Part-4" src="https://github.com/user-attachments/assets/0b468a8e-8651-41eb-80b9-a1470c063d76" />


The model achieved a very low MAE, indicating high accuracy in tracking temperature trends. The performance was visualized by plotting the predicted vs. actual temperature curves. The model was saved as `temperature_prediction_mlp.h5`.

---

## 6. Bonus: Single Neuron Weight Correction

### 6.1 Diagnosis and Fix

The original `Listing10_4.py` contained a shape mismatch in the weight initializer. In Keras, a `Dense` layer with 2 inputs and 1 output requires a kernel shape of `(2, 1)`.

* **Change:** We corrected the weight array from a 1D list `[.5, -0.5]` to a 2D list `[[0.5], [-0.5]]`.
* **Visualization:** The corrected script successfully produced a 3D surface plot, illustrating the "S-curve" decision boundary created by the Sigmoid activation function across a 2D input space.


<img width="1000" height="700" alt="Listing10_4_figure" src="https://github.com/user-attachments/assets/0510bc27-6ed0-4bc6-b51d-c381924ac6c1" />

---

## 7. Conclusion

This homework demonstrated the importance of the **"Train-Normalize-Deploy"** workflow. By using custom data parsers (Part 1), frequency-domain log-scaling (Part 2), and shape descriptors (Part 3), we proved that complex signals can be compressed into small, efficient MLP models. All four applications were successfully implemented, trained, and exported in the standard **.h5** format for future use.

