# Embedded MFCC Feature Extraction for Audio Signals

This project implements a complete, end-to-end audio feature extraction pipeline on an STM32F746NG Discovery board. The system is designed to compute Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio signals, converting large audio files into compact, information-rich feature vectors suitable for machine learning applications like keyword spotting or voice command recognition.

The project operates in a bare-metal Mbed OS environment, ensuring minimal resource overhead and high performance. It uses a host-client model, where a Python script on a PC sends audio data to the STM32 board, which performs the computation and returns the results.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Hardware and Software Requirements](#hardware-and-software-requirements)
- [System Architecture](#system-architecture)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Communication Protocol](#communication-protocol)

## Project Overview

The core goal of this project is to demonstrate an efficient implementation of the MFCC algorithm on a resource-constrained microcontroller. By offloading the computationally intensive feature extraction task to the embedded device, this system serves as a foundation for on-device machine learning, where data can be processed locally without needing to stream it to the cloud.

The pipeline includes:
1.  **Audio Data Transmission:** A Python script sends `.wav` files to the board via serial.
2.  **Embedded Signal Processing:** The STM32 board computes a 13-coefficient MFCC vector from the raw audio.
3.  **Optimized Computation:** The implementation leverages the **ARM CMSIS-DSP library** for high-performance FFT and signal processing functions.
4.  **Feature Return:** The board sends the computed MFCC vector back to the Python script for logging and analysis.

## Features
- **Bare-metal Mbed OS 6 Implementation:** Runs with minimal overhead for maximum performance.
- **End-to-End Pipeline:** Handles everything from reading `.wav` files to generating MFCC feature vectors.
- **ARM CMSIS-DSP Integration:** Uses hardware-accelerated libraries for optimized FFT calculations.
- **Host-Client Architecture:** A Python script automates testing and data collection.
- **Data Reduction:** Demonstrates a >95% reduction in data size from raw audio to MFCC features.
- **High-Quality Features:** The extracted features are shown to be highly discriminative for classification tasks.

## Hardware and Software Requirements

### Hardware
- **STM32F746NG Discovery Board**
- USB-A to Micro-USB cable

### Software
- **Mbed Studio:** For compiling and flashing the embedded application.
- **Python 3.x:** For running the host script.
- **Required Python Library:**
  ```bash
  pip install pyserial
  ```

## System Architecture

The system is composed of two main components:
1.  **Embedded Application (Client):** A C++ program running on the STM32. It listens for commands, receives audio data, performs MFCC calculations, and sends back the results.
2.  **Host Script (Host):** A Python script running on a PC. It reads audio files, sends them to the STM32, and logs the returned feature vectors.

This architecture allows for easy testing and batch processing of large audio datasets.

## File Structure

The project is organized for clarity and modularity:

```
.
├── mbed-os/                # Mbed OS library
├── external/               # External libraries
│   └── CMSIS_5/            # ARM CMSIS-DSP library
├── mfcc.cpp                # MFCC implementation logic
├── mfcc.h                  # MFCC header file
├── main.cpp                # Core application logic and serial communication
├── mbed_app.json           # Mbed project configuration
└── script.py          # Python script to communicate with the board
```

- **`main.cpp`**: Handles the serial communication protocol and orchestrates the feature extraction workflow.
- **`mfcc.cpp` / `mfcc.h`**: A self-contained module implementing the MFCC calculation based on standard algorithms.
- **`external/`**: Contains the ARM CMSIS-DSP library, manually integrated for optimized math functions.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Set up the Embedded Project:**
    - Open the project in **Mbed Studio**.
    - Ensure the **STM32F746NG Discovery board** is selected as the build target.
    - Compile the project and flash it to your board using the "Run program" button.

3.  **Set up the Python Environment:**
    - Install the required `pyserial` library:
      ```bash
      pip install pyserial
      ```
    - Connect the flashed STM32 board to your PC. Identify its serial port (e.g., `COM3`, `/dev/ttyACM0`).
    - Update the serial port in the host Python script and run it to begin sending audio data and collecting MFCC features.

## Communication Protocol

The communication between the Python script and the STM32 board is a simple, text-based serial protocol:

1.  **Host Sends Command:** The Python script sends a start command (e.g., `SEND_AUDIO\n`).
2.  **Host Sends Header:** The script sends metadata, such as the number of audio samples.
3.  **Host Sends Payload:** The script sends the raw audio bytes.
4.  **Board Computes:** The STM32 receives the data and computes the MFCCs.
5.  **Board Responds:** The board sends back the computed MFCC coefficients, formatted as text lines (e.g., `Coeff 0: -2.642317`).
6.  **End of Transmission:** The board sends a unique token (`[END_OF_FEATURES]\n`) to signal that the entire vector has been sent. The Python script listens for this token to complete the transaction.