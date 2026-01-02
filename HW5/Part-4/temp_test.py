import serial
import serial.tools.list_ports
import struct
import time
import pandas as pd
import numpy as np
import glob
import os

# --- CONFIGURATION ---
DATA_FOLDER = "temp_data/"
BAUD_RATE = 115200
TIMEOUT = 5.0
NUM_FEATURES = 22  # Columns 3 to 24 from your text file

def find_stm32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in (p.manufacturer or ""):
            return p.device
    return None

def load_and_preprocess_data(file_path):
    """
    Parses the SML2010 text format.
    Skips Date/Time and extracts the 22 numeric features.
    """
    print(f"Reading {file_path}...")
    # Load space-separated data, ignore lines starting with '#'
    df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    
    # Column 0: Date, Column 1: Time -> We skip these
    # Column 2 to 23 -> The 22 numeric features the MCU expects
    numeric_features = df.iloc[:, 2:].values.astype(np.float32)
    
    return numeric_features

def main():
    # 1. Setup Serial
    port = find_stm32_port()
    if not port:
        print("❌ Error: STM32 Board not found!")
        return

    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    
    # 2. Find Data Files
    files = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
    if not files:
        print(f"❌ Error: No .txt files found in {DATA_FOLDER}")
        return

    print(f"🔌 Connected to {port}")
    print(f"🚀 Starting Temperature Regression Test...")
    print("-" * 60)
    print(f"{'FILE':<15} | {'ROW':<5} | {'PREDICTION'}")
    print("-" * 60)

    for fpath in files:
        data_rows = load_and_preprocess_data(fpath)
        filename = os.path.basename(fpath)

        for i, row in enumerate(data_rows):
            # --- Handshake: Wait for [READY] ---
            # We clear the input buffer to ensure we catch a NEW signal
            ser.reset_input_buffer()
            ready = False
            t0 = time.time()
            while time.time() - t0 < 3.0:
                line = ser.readline().decode(errors='ignore').strip()
                if "[READY]" in line:
                    ready = True
                    break
            
            if not ready:
                print(f"Row {i}: MCU not responding. Check connection.")
                break

            # --- Send 22 Floats (88 bytes) ---
            # IMPORTANT: If your model was trained with a Scaler (StandardScaler/MinMax),
            # apply it here before sending: row = my_scaler.transform(row.reshape(1,-1))
            payload = struct.pack(f'<{NUM_FEATURES}f', *row)
            ser.write(payload)

            # --- Wait for Prediction ---
            prediction_msg = "---"
            t0 = time.time()
            while time.time() - t0 < TIMEOUT:
                line = ser.readline().decode(errors='ignore').strip()
                if "PREDICTED TEMPERATURE:" in line:
                    prediction_msg = line.split(":")[1].strip()
                    break
            
            print(f"{filename[:15]:<15} | {i:<5} | {prediction_msg}")

            # Optional: Add a tiny delay if you want to watch the logs
            # time.sleep(0.05)

    ser.close()
    print("-" * 60)
    print("Test Complete.")

if __name__ == "__main__":
    main()