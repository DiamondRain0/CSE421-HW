import serial, time, csv, sys
import numpy as np
import cv2
from sklearn import datasets
from datetime import datetime

BAUD = 115200
TIMEOUT = 10.0
NTEST = 200
CSV_FILE = "inference_results.csv"

def load_mnist():
    print("Loading MNIST...")
    mnist = datasets.fetch_openml("mnist_784", version=1)
    X = mnist.data.astype(np.uint8).to_numpy()
    y = mnist.target.astype(int).to_numpy()
    return X[-NTEST:], y[-NTEST:]

def to_32x32x3(img):
    img = img.reshape(28,28)
    # MNIST is white on black; ensure your model was trained on this polarity
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_NEAREST)
    img = np.stack([img,img,img], axis=-1)
    return img.astype(np.uint8)

# 1. Initialize Serial
ser = serial.Serial("COM3", BAUD, timeout=0.1)
time.sleep(2) # Wait for STM32 reset

X, y = load_mnist()
correct = 0

# 2. Setup CSV and Start Loop
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["Timestamp", "Index", "Actual", "Predicted", "Match"])

    for i in range(NTEST):
        # Wait for [READY] signal from STM32
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if "[READY]" in line:
                    break
        else:
            print(f"READY timeout at index {i}")
            continue

        # Prepare and send image (3072 bytes)
        img = to_32x32x3(X[i])
        ser.write(img.tobytes())
        ser.flush()

        # Wait for RESULT
        pred = None
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                # Check for RESULT or PREDICTED_CLASS based on your C++ code
                if "RESULT" in line or "PREDICTED_CLASS" in line:
                    try:
                        pred = int(line.split(":")[1].strip())
                        break
                    except:
                        continue
        
        if pred is not None:
            is_match = (pred == y[i])
            correct += is_match
            acc = 100 * correct / (i+1)
            
            # Log to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, i, y[i], pred, int(is_match)])
            
            print(f"IDX {i:03d} | REAL {y[i]} | PRED {pred} | ACC {acc:.2f}%")
        else:
            print(f"RESULT timeout at index {i}")

ser.close()
print(f"Testing complete. Results saved to {CSV_FILE}")