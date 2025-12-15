import serial
import serial.tools.list_ports
import struct
import time
import numpy as np
import pandas as pd
import sys
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "./Part-2/WISDM_ar_v1.1_raw.txt"
BAUD_RATE = 115200
TIMEOUT = 2.0
TEST_SAMPLES = 500  # Target number of samples
WINDOW_SIZE = 80
STEP_SIZE = 40

# ==========================================
# DATA LOADING
# ==========================================
def parse_data(path):
    print(f"Loading data from: {os.path.abspath(path)}")
    if not os.path.exists(path):
        print("❌ Error: File not found.")
        return None

    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                try:
                    # Format: 33,Jogging,491059...,-0.69,12.68,0.50;
                    parts = line.strip().rstrip(';').split(',')
                    if len(parts) == 6:
                        # [user, activity, timestamp, x, y, z]
                        data.append([int(parts[0]), parts[1], float(parts[3]), float(parts[4]), float(parts[5])])
                except: continue
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    return pd.DataFrame(data, columns=['user', 'act', 'x', 'y', 'z'])

def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in p.manufacturer or "ST-Link" in p.description:
            return p.device
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Load Data
    df = parse_data(DATA_PATH)
    if df is None or df.empty: return

    # Filter for Test Users (> 28)
    df_test = df[df['user'] > 28].reset_index(drop=True)
    if df_test.empty:
        print("❌ Error: No test data found (User IDs > 28).")
        return

    # 2. Prepare Random Indices
    print("Indexing valid windows...")
    valid_start_indices = list(range(0, len(df_test) - WINDOW_SIZE, STEP_SIZE))
    
    # --- FIX: Use a local variable instead of modifying the global ---
    n_samples = TEST_SAMPLES
    
    if len(valid_start_indices) < n_samples:
        print(f"⚠️ Warning: Requested {n_samples} samples, but only {len(valid_start_indices)} available.")
        n_samples = len(valid_start_indices)

    # SHUFFLE to test randomly
    random.shuffle(valid_start_indices)
    selected_indices = valid_start_indices[:n_samples]

    # 3. Connect to MCU
    port = find_port()
    if not port: 
        print("❌ STM32 Board not found.")
        return
    
    print(f"🔌 Connecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
    except:
        print("❌ Connection Failed.")
        return

    # 4. Test Loop
    print(f"\n🚀 Starting Random Test ({n_samples} samples)...")
    print(f"{'#':<4} | {'REAL ACT':<10} | {'PRED':<5} | {'RES':<4} | {'ACCURACY':<8}")
    print("-" * 50)

    correct_count = 0
    total_processed = 0

    for i, start_idx in enumerate(selected_indices):
        # Extract Segment
        segment = df_test.iloc[start_idx : start_idx + WINDOW_SIZE]
        
        # Ground Truth
        real_act = segment['act'].iloc[0] # Use label of first sample
        real_bin = 1 if real_act == 'Walking' else 0
        
        # Prepare Payload (X array, Y array, Z array)
        xs = segment['x'].values.astype(np.float32)
        ys = segment['y'].values.astype(np.float32)
        zs = segment['z'].values.astype(np.float32)
        
        payload = b''
        payload += struct.pack(f'<{WINDOW_SIZE}f', *xs)
        payload += struct.pack(f'<{WINDOW_SIZE}f', *ys)
        payload += struct.pack(f'<{WINDOW_SIZE}f', *zs)

        # --- A. Handshake (Wait for Ready) ---
        ready = False
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    if "[READY]" in line: 
                        ready = True; break
                except: pass
        
        if not ready:
            print(f"⚠️ Timeout waiting for Ready at sample {i}")
            ser.reset_input_buffer()
            continue

        # --- B. Send Data ---
        ser.write(payload)

        # --- C. Read Prediction ---
        pred = -1
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    if "PREDICTION:" in line:
                        # Parse "PREDICTION: 1 (Prob: ...)"
                        pred = int(line.split(':')[1].split('(')[0].strip())
                        break
                except: pass
        
        # --- D. Stats ---
        total_processed += 1
        is_correct = (pred == real_bin)
        
        if is_correct: 
            correct_count += 1
            icon = "✅"
        else:
            icon = "❌"

        # Calculate Running Accuracy
        current_acc = (correct_count / total_processed) * 100.0
        
        # Print Status
        print(f"{i+1:<4} | {real_act[:9]:<10} | {pred:<5} | {icon:<4} | {current_acc:.1f}%")

    # Final Result
    print("-" * 50)
    print(f"🎉 DONE. Final Accuracy: {current_acc:.2f}%")
    ser.close()

if __name__ == "__main__":
    main()