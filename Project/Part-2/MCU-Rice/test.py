import serial
import time
import csv
import os
import numpy as np
import cv2
from datetime import datetime

# ================================
# CONFIG
# ================================
PORT = "COM3"
BAUD = 115200
TIMEOUT = 10           # seconds for READY / RESULT
NTEST = 100
CSV_FILE = "rice_stm32_results.csv"
DATA_DIR = "Rice_Image_Dataset"
IMG_SIZE = 96

# ================================
# 1. Load Rice Images
# ================================
X, y = [], []
class_names = sorted(os.listdir(DATA_DIR))

for idx, cls in enumerate(class_names):
    cls_dir = os.path.join(DATA_DIR, cls)
    for f in os.listdir(cls_dir):
        path = os.path.join(cls_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(idx)

X = np.array(X, dtype=np.uint8)
y = np.array(y, dtype=np.int32)

# Shuffle dataset
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

print(f"Classes: {class_names}")
print(f"Total test images: {len(X)}")

# ================================
# 2. Connect to STM32
# ================================
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)  # give STM32 time to reset
except Exception as e:
    raise RuntimeError(f"Failed to open serial port {PORT}: {e}")

correct = 0

# ================================
# 3. Test Loop
# ================================
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Index", "True", "Pred", "Match"])

    for i in range(min(NTEST, len(X))):
        # ----- Wait for [READY] -----
        ready = False
        t_start = time.time()
        while time.time() - t_start < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if "[READY]" in line:
                    ready = True
                    break
        if not ready:
            print(f"{i:03d} | READY timeout")
            continue

        # ----- Send image bytes -----
        try:
            ser.write(X[i].tobytes())
            ser.flush()
        except Exception as e:
            print(f"{i:03d} | ERROR sending image: {e}")
            continue

        # ----- Wait for RESULT -----
        pred = None
        t_start = time.time()
        while time.time() - t_start < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line.startswith("RESULT:"):
                    try:
                        pred = int(line.split(":")[1])
                        break
                    except ValueError:
                        continue

        if pred is None:
            print(f"{i:03d} | RESULT timeout")
            continue

        # ----- Update stats -----
        match = int(pred == y[i])
        correct += match
        acc = 100 * correct / (i + 1)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([ts, i, class_names[y[i]], class_names[pred], match])

        print(f"{i:03d} | TRUE {class_names[y[i]]:<10} | PRED {class_names[pred]:<10} | ACC {acc:.2f}%")

# Close serial safely
ser.close()
print(f"Testing finished → {CSV_FILE}")
