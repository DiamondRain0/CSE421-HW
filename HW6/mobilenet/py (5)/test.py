import serial
import time
import csv
import sys
import numpy as np
import cv2
from sklearn import datasets
from datetime import datetime

# Constants matching C++ and Notebook
BAUD = 115200
TIMEOUT = 10.0
NTEST = 200
CSV_FILE = "inference_results.csv"
LOG_FILE = "mcu_debug_log.txt"  # <--- NEW: File for all MCU prints
IMG_W, IMG_H, IMG_C = 32, 32, 3
EXPECTED_BYTES = IMG_W * IMG_H * IMG_C 

def load_mnist():
    print("Loading MNIST via sklearn...")
    mnist = datasets.fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.uint8)
    y = mnist.target.astype(int)
    return X[-NTEST:], y[-NTEST:]

def preprocess_for_stm32(img_flat):
    """Replicates the notebook preprocessing: Resize 28x28 -> 32x32 and stack 3 channels"""
    img = img_flat.reshape(28, 28)
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_rgb = np.stack([img_resized] * 3, axis=-1)
    return img_rgb.astype(np.uint8)

# Initialize Serial
try:
    ser = serial.Serial("COM3", BAUD, timeout=0.1)
    time.sleep(2) 
    ser.reset_input_buffer()
except Exception as e:
    print(f"Error opening serial port: {e}")
    sys.exit(1)

X_raw, y = load_mnist()
correct = 0

# Open both the CSV for results and the TXT for raw prints
with open(CSV_FILE, mode='w', newline='') as f_csv, \
     open(LOG_FILE, mode='w') as f_log: # <--- NEW: Open log file
    
    writer = csv.writer(f_csv)
    writer.writerow(["Timestamp", "Index", "Actual", "Predicted", "Match"])
    
    f_log.write(f"--- Session Started: {datetime.now()} ---\n")

    print(f"Starting test. Logging prints to {LOG_FILE}...")
    
    for i in range(NTEST):
        img_to_send = preprocess_for_stm32(X_raw[i])
        img_bytes = img_to_send.tobytes()

        # 1. Wait for [READY] signal
        t0 = time.time()
        ready_received = False
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                
                # NEW: Save every line to the log file regardless of content
                if line:
                    f_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] MCU: {line}\n")
                    f_log.flush() # Ensure it writes to disk immediately
                
                if "[READY]" in line:
                    ready_received = True
                    break
        
        if not ready_received:
            print(f"IDX {i}: [READY] timeout")
            continue

        # 2. Send image data
        ser.write(img_bytes)
        ser.flush()

        # 3. Wait for PREDICTED_CLASS
        pred = None
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                
                # NEW: Save output prints to log file
                if line:
                    f_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] MCU: {line}\n")
                    f_log.flush()

                if "PREDICTED_CLASS" in line:
                    try:
                        pred = int(line.split(":")[1].strip().split()[0])
                        break
                    except (IndexError, ValueError):
                        continue
        
        # 4. Finalize result logic
        if pred is not None:
            is_match = (pred == y[i])
            correct += int(is_match)
            acc = 100 * correct / (i + 1)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, i, y[i], pred, int(is_match)])
            print(f"IDX {i:03d} | REAL {y[i]} | PRED {pred} | ACC {acc:.2f}%")
        else:
            print(f"IDX {i:03d} | RESULT timeout")

ser.close()
print(f"Testing complete. Logs saved to {LOG_FILE}")