# CSE421 Homework 1 Report

## Part-1(Reading Temperature Value From MCU)

### 1.1. Overview

This section details the implementation of a bare-metal application on the STM32F746NG
Discovery board to read its internal temperature sensor and convert the raw
Analog-to-Digital Converter (ADC) values into degrees Celsius. The project leveraged a
modified version of the MBED Studio Blinky example as a starting point.

### 1.2. Implementation

The core of this implementation involved configuring the STM32's internal ADC to sample
the temperature sensor and subsequently converting the raw digital readings into a
meaningful Celsius temperature. A key challenge encountered was the inability to directly
display floating-point numbers via serial communication in the default bare-metal setup. This
was resolved by explicitly enabling floating-point support within the mbed_app.json
configuration file, allowing the processed temperature values to be accurately transmitted
and displayed.

<img width="1226" height="662" alt="image" src="https://github.com/user-attachments/assets/88d906b3-bbda-4206-8490-7c31e1196a3e" />

Figure 1.1: mbed_app.json


#### 1.2.1. Analog Input Declaration

The internal temperature sensor is accessed through the microcontroller's Analog-to-Digital
Converter (ADC). In Mbed OS, this is simplified using the AnalogIn class:

  - AnalogIn is an Mbed OS class used to instantiate an object (intTemp) for interfacing
with analog input pins or internal ADC channels.
  - ADC_TEMP is a predefined constant specific to the STM32 microcontroller, routing
the internal temperature sensor's output to an available ADC channel.
  - The intTemp object allows the program to read the raw analog voltage (Vsense)
produced by the internal sensor.

#### 1.2.2. Main Loop

The core logic resides within an infinite loop, continuously acquiring and processing
temperature data:

  - This while (true) loop ensures continuous operation, repeatedly reading and
processing temperature data.
  - The intTemp.read_u16() method samples the internal temperature sensor and
returns a 16-bit unsigned integer (ranging from 0 to 65535), representing the digital
conversion of the analog voltage at the sensor.

#### 1.2.3. Convert ADC Value to Voltage

To interpret the raw ADC value, it must be converted back to a voltage. Given that the
STM32's ADC reference voltage is typically 3.3V, the conversion is performed as follows:

<img width="984" height="245" alt="image" src="https://github.com/user-attachments/assets/ed6234ff-255b-4d58-8968-6e60bf1cf243" />

Figure 1.2: Converting ADC value to voltage

  - The STM32's ADC typically uses a 3.3 V reference voltage.
  - This equation linearly scales the raw 16-bit ADC reading (value) to its corresponding
voltage (Vsense), accounting for the full-scale ADC range (0-65535) and the
reference voltage (3.3V).


#### 1.2.4. Convert Voltage to Temperature

Finally, the sensed voltage (Vsense) is converted into a temperature in Celsius using the
sensor's calibration data, typically provided in the STM32 datasheet:

<img width="1036" height="242" alt="image" src="https://github.com/user-attachments/assets/9fb4a58b-ca45-47ff-a46c-735730f658ca" />

Figure 1.3: Converting voltage to temperature
  
  - This formula is derived from typical STM32 internal temperature sensor calibration
data:
    - Voltage at 25°C (V_25): Approximately 0.76 V
    - Sensor slope (Avg_Slope): Approximately 2.5 mV/°C (or 0.0025 V/°C)
  - The linear relationship Temperature = ((Vsense - V_25) / Avg_Slope) + 25.0f is used
to calculate the approximate temperature in degrees Celsius.

<img width="1298" height="938" alt="carbon" src="https://github.com/user-attachments/assets/ba23484d-2927-4991-9a5f-06ad75965a75" />

Figure 1.4: main.cpp


### 1.3. Results
<img width="692" height="1012" alt="carbon (1)" src="https://github.com/user-attachments/assets/53005038-372d-4fa3-be71-03f874d8ce95" />

Figure 1.5: Result of Part-1


## Part-2 (Application: Feature Extraction from Audio

## Signals)

### 2.1. Implementation Overview

In this project, we implemented a complete audio feature extraction pipeline on a
microcontroller, designed to compute Mel-Frequency Cepstral Coefficients (MFCCs) from
raw audio signals. The system was developed on a bare-metal Mbed OS 6 environment,
ensuring minimal resource overhead and high performance.

#### 2.1.1 File Structure

The project was organized for modularity and clarity:

  - main.cpp: Core application handling serial communication and orchestrating the
feature extraction workflow.
  - mfcc.h & mfcc.cpp: Self-contained module implementing the MFCC calculation logic
based on standard textbooks.
  - external/: Directory containing the ARM CMSIS-DSP library, manually integrated to
provide optimized FFT and signal processing functions.

#### 2.1.2 Communication Protocol

A serial link (COM6) connected the microcontroller to a host PC. A Python script on the PC
sent .wav file data to the microcontroller, which processed the audio and returned a
formatted MFCC vector. All responses were logged to a text file for later analysis.

#### 2.1.3 Conceptual Background: The Role and Meaning of MFCCs

The transformation of a high-dimensional, raw audio waveform into a low-dimensional,
information-rich feature vector using Mel-Frequency Cepstral Coefficients (MFCCs) is a
foundational step in speech processing. The process is engineered to mimic human auditory
perception and isolate the phonetic components of speech. The key concepts are as follows:

  - Mel-Frequency Scale: This component models the non-linear nature of human
hearing, which is more sensitive to changes in lower frequencies. The audio
spectrum is warped using a Mel filterbank, which emphasizes frequencies below 1
kHz—where critical information for distinguishing vowels resides—while compressing
higher frequencies. This ensures the features are based on the most perceptually
relevant parts of the signal.
  - Cepstral Analysis (via Discrete Cosine Transform - DCT): After obtaining the
log-energies from the Mel filterbank, the DCT is applied. This transform serves two
vital functions:
    - It decorrelates the energies from adjacent Mel filters, which is a highly
desirable property for many machine learning algorithms.
    - It compacts the energy, effectively separating the broad shape of the log-Mel
spectrum from its finer details into the first few coefficients.
  - Coefficient Interpretation: Each of the resulting coefficients carries specific,
meaningful information about the sound:
    - C0 (Log-Energy): Represents the overall log-energy or loudness of the audio
frame. While useful for detecting the presence of speech, it is often less
critical for classification, as the same word can be spoken at different
volumes.
    - Low-Order Coefficients (approx. C1-C3): These are arguably the most critical
for speech recognition. They capture the gross shape of the spectral
envelope, including the overall "spectral tilt" and the location of major
resonant peaks (formants). As this shape is dictated by the physical
configuration of the human vocal tract, these coefficients are extremely
effective at distinguishing between different vowels (e.g., 'ah' vs. 'ee').
    - High-Order Coefficients (approx. C4-C12): These represent the finer details
and texture of the spectral envelope. They are essential for distinguishing
between consonants, which are often defined by quick, intricate acoustic
events like the hissing of an 's' or the transient burst of a 'p'.
In essence, the MFCC pipeline intelligently reduces thousands of raw audio samples into a
compact, 13-coefficient vector. This vector serves as a robust "fingerprint" of the sound,
retaining critical phonetic information while discarding redundant data. This high degree of
data reduction and information preservation makes MFCCs an ideal feature set for efficient
and accurate classification on resource-constrained microcontrollers.


#### 2.1.4 Example Output

<img width="431" height="767" alt="image" src="https://github.com/user-attachments/assets/21dd3cb2-97f4-4683-9f4b-f59a6b85e309" />

Figure 2.1: Example Output


### 2.2. Detailed Results and Analysis: The Power of Feature Extraction

The purpose of feature extraction in an embedded context is twofold:

  - Data reduction — dramatically reduce the amount of data to store and process.
  - Information preservation — ensure the extracted features retain sufficient information
for machine learning tasks.
  - Our results demonstrate success on both fronts.

#### 2.2.1 Quantifying the Benefit: Drastic Data Reduction

Raw audio files are prohibitively large for microcontrollers. Converting raw audio to a
13-coefficient MFCC vector achieves dramatic size reduction:

<img width="925" height="685" alt="image" src="https://github.com/user-attachments/assets/9db04015-a615-4085-ba36-86d921afdcac" />

Figure 2.2: Histogram of the Feature Size / Audio Size ratio across all processed samples.
The peak of the distribution is between 0.025 and 0.05, meaning MFCC features are 2.5% to
5% of the original audio size.
This corresponds to a 95%–97.5% reduction, freeing RAM and reducing computational load
— critical for real-time embedded processing.


#### 2.2.2 Verifying the Quality: Are the Features Meaningful?

Data reduction alone is insufficient if the features cannot distinguish between classes. We
validated the MFCCs using spoken digits “0” and “1”.
Principal Component Analysis
We used Principal Component Analysis (PCA) to reduce the 13-dimensional MFCC vectors
to 2 dimensions for visualization:

<img width="942" height="688" alt="image" src="https://github.com/user-attachments/assets/f17a4ef6-fa19-45fe-970d-516b2e38a886" />

Figure 2.3: 2D PCA of MFCC vectors. Blue = Digit 0, Orange = Digit 1.
Distinct clusters demonstrate that MFCC features are highly separable.
Clear separation indicates that a machine learning model can reliably classify digits based
on these features.


Mean MFCC Fingerprint
Plotting the mean MFCC vector for each digit, with shaded standard deviation:

<img width="958" height="545" alt="image" src="https://github.com/user-attachments/assets/f1af7361-83b9-4c98-8ad8-609ef965ae0f" />

Figure 2.4: Mean MFCC fingerprints.
The trajectories show distinct spectral patterns for each digit.
This summary reinforces that MFCC features capture unique, consistent acoustic signatures.


Individual MFCC Coefficients
To understand the source of this separability, we can examine the distribution of each
coefficient individually. The histograms in Figure 6 clearly show that the distributions for Digit
0 and Digit 1 differ significantly for several coefficients. For example, in coefficients C1, C2,
and C3, there is a clear offset between the central tendencies of the two distributions. These
statistical differences across multiple coefficients are what allow a machine learning model to
effectively distinguish between the two spoken digits.

<img width="974" height="648" alt="image" src="https://github.com/user-attachments/assets/f73e0c02-9e95-4eb6-82f6-460415ac4fcd" />

Figure 2.5: Distribution histograms for each of the 13 MFCC coefficients.


### 2.3. Conclusion

This project demonstrates a complete end-to-end embedded MFCC extraction pipeline:
Massive Data Reduction: MFCC features occupy only 2.5–5% of raw audio size, enabling
resource-efficient processing.
High-Quality Features: PCA, boxplots, and mean fingerprint plots confirm that features are
discriminative and suitable for classification.
The system is now validated and can serve as the foundation for on-device machine
learning, such as real-time keyword spotting.


