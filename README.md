<img width="1565" height="343" alt="image" src="https://github.com/user-attachments/assets/1096073b-0aa6-4b2b-b17d-9f1afd59f1c5" />

# CSE421 — Embedded Machine Learning Homework Collection

This repository is the course deliverable for CSE421: a collection of embedded signal-processing
and machine-learning projects developed for the STM32F7 Discovery family. The work demonstrates
end-to-end pipelines: dataset preparation and training in Python, model conversion/quantization,
and resource-aware inference on an STM32F746G-DISCO (Cortex-M7, FPU).

**Status:** Each homework (HW1..HW6) is a self-contained project with reports, firmware, host-side
tools, and conversion artifacts. See the per-homework READMEs for deep details.

**Table of Contents**
- **Overview**
- **Supported Hardware & Prerequisites**
- **Repository Structure**
- **Quick Start (Build & Run)**
- **HIL Communication Protocol**
- **Model Conversion & Quantization**
- **Testing & Validation**
- **Contributing**
- **Licenses & Attribution**

**Overview**
- Purpose: Provide reproducible examples of deploying classical ML and small neural networks
  to resource-constrained MCUs. Key topics: MFCC extraction, Bayes/k-NN/SVM, MLPs/CNNs,
  TFLite quantization, and MCU-side inference harnesses.

**Supported Hardware & Prerequisites**
- **Target board:** STM32F746G-DISCO (DISCO_F746NG). All projects target an FPU-capable Cortex-M7.
- **Host tools (recommended):** Python 3.8+, numpy, scipy, scikit-learn, tensorflow, librosa, matplotlib.
- **MCU toolchains & IDEs:**
  - **HW1:** developed and built in **Mbed Studio** (see `HW1/HW1-part1/mbed_app.json`).
  - **HW2..HW6 and Project:** developed and built using **Keil Studio integrated with VS Code** (Keil + VS Code extensions). Ensure the Keil/ARM toolchain, the VS Code C/C++ extensions, and Keil integration are installed and configured. For float-based builds enable hardware FPU support in project settings.



**Repository Structure**
- **HW1/** — Temperature sensor demo (Part-1) and MFCC feature-extraction firmware (Part-2).
  Minimal MCU demonstration was implemented with **Mbed Studio**. See `HW1/HW1-part1/mbed_app.json` for FPU config and `HW1/HW1-part2/script.py` for the host-side MFCC streaming tool.
- **HW2/** — Classical ML pipeline: Bayes (QDA), k-NN, and SVM. Contains training notebooks, `sklearn2c`-style model exports, and Keil/VS Code MCU projects for inference and HIL validation. See `HW2/Readme.md` for architecture and comms flowchart.
- **HW3/** — Lightweight time-series regression (temperature forecasting) and a compact HAR single-neuron classifier. Firmware and host tests are Keil/VS Code projects; notebooks and MCU glue live in `HW3/`.
- **HW4/** — Focused MLP work: Human Activity Recognition, Keyword Spotting, MNIST (Hu moments), and temperature regression. This folder contains training notebooks, exported `.h5` models (e.g., `Part-1/mlp_har_model.h5`), and conversion notes. HW4 is a strong example of end-to-end model design → quantization → MCU integration using Keil/VS Code.
- **HW5/** — MCU test harnesses, TFLite verification, and validation scripts that exercise the deployed models across HAR, MFCC, MNIST, and temperature tasks. Keil/VS Code builds and host-side test drivers are included.
- **HW6/** — Comparative CNN experiments (Custom CNN, EfficientNet, MobileNet, ResNet, SqueezeNet) with full quantization and MCU deployment pipeline. This is one of the most complete sections for production-style optimization and benchmarking; results are saved under `HW6/results/` and per-architecture subfolders include notebooks and conversion scripts.
- **Project/** — Independent project work and capstone-like experiments (e.g., FOMO, Rice classifier, regression studies). These are Keil/VS Code based MCU integrations plus supporting training notebooks and dataset artifacts.

**Project (Detailed)**

The `Project/` folder contains the repository's capstone-style work. Two subprojects are the primary contributions: **Part-1 (FOMO)** and **Part-2 (Rice classifier)**. These two parts combine end-to-end model development, quantization/optimization, and Keil/VS Code MCU deployment with Hardware-in-the-Loop validation.

- **Part-1 — FOMO (Project/Part-1)**
  - Objective: build and validate a compact MCU classifier (FOMO) that detects a specified event or pattern from sensor-derived features.
  - Contents: `Project/Part-1/MCU_FOMO/` (firmware and Keil project files), `Project/Part-1/Train_FOMO/` (training notebooks and conversion scripts), and `Project/Part-1/Results-FOMO/` (validation logs, CSVs, and accuracy results).
  - Training & conversion: The offline workflow trains a reference model in Python (notebooks in `Train_FOMO`), then converts and quantizes the model (TFLite INT8 or C export) using a representative dataset generator. Conversion scripts and calibration datasets reside alongside the training notebooks.
  - MCU integration: The MCU project in `MCU_FOMO` is organized for Keil/VS Code. Large model arrays are declared `const` to place them in Flash. The Keil project uses a TensorArena or a small inference routine depending on whether TFLite Micro or a hand-written inference function is used.
  - How to run: create a Keil project using the Blinky template (see Quick Start), copy the files from `Project/Part-1/MCU_FOMO` into the Blinky project `src/` folder, build, and flash. Host-side validation scripts in `Results-FOMO/` show the expected serial protocol and CSV output format.
  - Inspiration & attribution: Project Part-1 (FOMO) was inspired by and references the upstream FOMO repository: https://github.com/bhoke/FOMO — consult that project for additional design ideas and reference implementations.

- **Part-2 — Rice Classifier (Project/Part-2)**
  - Objective: classify rice grain types (or other agricultural varieties) using a compact model suitable for MCU deployment. This part demonstrates image/feature processing, lightweight model architecture design, and the full conversion/validation pipeline.
  - Contents: `Project/Part-2/MCU-Rice/` (Keil project and MCU inference code), `Project/Part-2/Train-Rice/` (training notebooks, preprocessing scripts), and `Project/Part-2/Result-Rice/` (evaluation outputs and inference logs).
  - Training & conversion: Training notebooks implement preprocessing (feature extraction or image resizing), model training, and representative dataset creation. The conversion step typically outputs a quantized `.tflite` and a C array for inclusion in the MCU project.
  - MCU integration: `MCU-Rice` contains the Keil/VS Code project configured for the `DISCO_F746NG` target. The source organizes preprocessing code, inference engine, and UART HIL handlers. The README inside `Project/Part-2` contains detailed build notes and expected serial packet formats for host testing.
  - How to run: copy the MCU source from `Project/Part-2/MCU-Rice` into a Keil Blinky project, ensure the project configuration matches the required heap/stack for model tensors, build, and flash. Use the host-side scripts in `Result-Rice` to stream test images/features and collect results.

Common notes for both parts:
  - Keil/VS Code: Both parts are designed for Keil Studio integrated into VS Code. Follow the Quick Start steps to install the Keil extension and create a Blinky-based Keil project; then copy the MCU sources into that project.
  - Representative data & calibration: Each `Train_*` folder contains the scripts to generate representative datasets required for post-training quantization — keep these consistent with the MCU preprocessing (scaler means/std, input order).
  - Validation: Results folders contain CSV logs and example serial transcripts. Use these to confirm parity between the Python reference implementation and MCU inference.

**Quick Start (Keil Studio on VS Code)**

This repository's MCU projects (HW2..HW6 and `Project/`) are developed for Keil Studio integrated
into Visual Studio Code. Follow these steps to create a new Keil project and reuse the MCU sources
from any homework.

1. Install the Keil Studio extension for VS Code:

  - Marketplace: https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack

2. Create a new Keil project using the Blinky pre-project template:

  - In VS Code, open the Keil Studio extension commands and select "Create Solution".
  - Select the target board: **STM32F746G-DISCO (DISCO_F746NG)**.
  - Choose the **Blinky** pre-project/template when prompted.


3. Copy the MCU firmware from a homework into your new project:

  - Locate the MCU sources for the homework you want to run (for example, `HW4/Part-1/` or `HW6/customCNN/mcu`).
  - Copy the C/C++ source files, headers, and any `mbed_app.json` or project config snippets into the Keil project folder (keep folder structure tidy under `src/` or `app/`).
  - Update the Keil project include paths and linker settings if required (especially if the homework requires extra heap/stack or places large arrays in Flash).

4. Build and flash

  - Use the Keil Studio build and flash commands from within VS Code. Confirm the target remains `DISCO_F746NG` and that hardware FPU is enabled for float-based projects.

This approach lets you reuse the MCU implementation from any homework by copying the MCU portion into a standard Blinky-keil project and building with Keil/VS Code.

**HIL Communication Protocol**
- A Stop-and-Wait UART handshake is used across many assignments:
  1. MCU sends `[READY]` to host.
  2. Host transmits one data chunk (raw samples, features, or an image).
  3. MCU replies `[SAMPLE_OK]` after reception.
  4. MCU runs feature extraction / inference and returns a label or vector.
- Packet formats vary by homework. See host scripts in each homework (e.g., [HW1/HW1-part2/script.py](HW1/HW1-part2/script.py))
  and the MCU `main.cpp` implementation for precise formats.

**Model Conversion & Quantization**
- Typical pipeline used in HW4/HW6:
  1. Train model in Python and save as `.h5`.
  2. Create a small representative dataset generator for calibration.
  3. Convert to TFLite using `tf.lite.TFLiteConverter` with `Optimize.DEFAULT`.
  4. Target INT8 quantization and produce a `.tflite` file.
  5. Generate C arrays for TFLite Micro and integrate into MCU project.

Example conversion snippet:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open('model_quant.tflite', 'wb').write(tflite_model)
```

**Memory & Performance Notes**
- Declare large constant arrays as `const` in C/C++ so they are placed in Flash, preserving SRAM.
- Increase linker heap/stack size when MFCC buffers or convolutional arenas require more memory.
- Enable hardware floating point in the project config when running floating-point inference.

**Testing & Validation**
- Many homeworks include host-side validation scripts that stream datasets and write CSV logs
  (see `HW6/results/` and `HW1/mfcc_results0.txt`). Use these to verify parity between
  Python reference outputs and MCU predictions.

**Contributing**
- To contribute: open an issue describing the change, then submit a PR against this repo.
- When adding firmware, preserve the folder layout and provide conversion scripts rather than
  committing large binary model blobs when possible.

**Licenses & Attribution**
- Several homework subfolders contain `LICENSE` files. Respect each subproject's license before reuse.

**Contact & Authors**
- Author and report metadata are included in each homework README. For questions about a specific
  homework, start by reading the per-homework README (linked in the Repository Structure).

