import serial
import serial.tools.list_ports
import struct
import time
import numpy as np
import pandas as pd
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = "test.csv"        
BAUD_RATE = 115200
TIMEOUT = 2.0

# !!! CRITICAL: Must match your training code !!!
WINDOW_SIZE = 80             
STEP_SIZE = 80               
MAX_WINDOWS = 500            

# LabelEncoder usually sorts alphabetically. 
# Double check your training results to see if this mapping is correct:
# 0: Downstairs, 1: Running, 2: Sitting, 3: Standing, 4: Upstairs, 5: Walking
LABELS_MAP = {
    0: "Downstairs",
    1: "Running",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Walking"
}

# ==========================================
# UTILITIES
# ==========================================
def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in (p.manufacturer or "") or "ST-Link" in p.description:
            return p.device
    return None

def load_csv(path):
    print(f"📂 Loading CSV: {os.path.abspath(path)}")
    if not os.path.exists(path):
        print("❌ CSV file not found")
        return None
    df = pd.read_csv(path)
    return df

# ==========================================
# MAIN
# ==========================================
def main():
    df = load_csv(CSV_PATH)
    if df is None or df.empty: return

    # Generate window indices
    indices = list(range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE))

    if MAX_WINDOWS is not None and len(indices) > MAX_WINDOWS:
        indices = random.sample(indices, MAX_WINDOWS)

    port = find_port()
    if not port:
        print("❌ STM32 board not found")
        return

    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"\n🚀 Starting HAR Test (Window Size: {WINDOW_SIZE})")
    print(f"{'#':<4} | {'REAL (ID)':<10} | {'PRED (ID)':<10} | {'MATCH':<5} | {'ACCURACY'}")
    print("-" * 65)

    correct = 0
    total = 0

    for i, start in enumerate(indices):
        segment = df.iloc[start:start + WINDOW_SIZE]
        
        # 1. Get Real Label (ensure it's an int)
        try:
            real_val = int(segment['Activity'].iloc[0])
        except ValueError:
            # If your CSV has string names, you need to map them back to IDs here
            real_val = segment['Activity'].iloc[0]

        # 2. Extract and Pack raw sensor data
        xs = segment['x-acc'].values.astype(np.float32)
        ys = segment['y-acc'].values.astype(np.float32)
        zs = segment['z-acc'].values.astype(np.float32)

        # Payload: [int32 N] followed by [N floats X], [N floats Y], [N floats Z]
        payload = struct.pack('<i', WINDOW_SIZE)
        payload += struct.pack(f'<{WINDOW_SIZE}f', *xs)
        payload += struct.pack(f'<{WINDOW_SIZE}f', *ys)
        payload += struct.pack(f'<{WINDOW_SIZE}f', *zs)

        # --- A. Synchronize ---
        ready = False
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if "[READY]" in line:
                    ready = True
                    break
        if not ready:
            print(f"[{i+1}] Timeout")
            continue

        # --- B. Send Data ---
        ser.write(payload)

        # --- C. Get Prediction ---
        pred_val = -1
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if "PREDICTED_CLASS:" in line:
                    try:
                        pred_val = int(line.split(":")[1].strip())
                    except:
                        pred_val = -1
                    break

        # --- D. Analysis ---
        total += 1
        # Match as integers
        is_correct = (pred_val == real_val)
        if is_correct:
            correct += 1
            icon = "✅"
        else:
            icon = "❌"

        current_acc = (correct / total) * 100.0
        
        # Display labels if they exist in our map
        real_name = LABELS_MAP.get(real_val, str(real_val))
        pred_name = LABELS_MAP.get(pred_val, str(pred_val))

        print(f"{i+1:<4} | {real_val} | {pred_val} | {icon:<5} | {current_acc:6.2f}%")

    print("-" * 65)
    print(f"🎉 TEST COMPLETE. Final Accuracy: {(correct/total)*100:.2f}%")
    ser.close()

if __name__ == "__main__":
    main()