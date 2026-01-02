import serial
import serial.tools.list_ports
import time
import struct
import numpy as np

# --- CONFIGURATION ---
# Update these paths to match your folder location
IMAGES_PATH = "MNIST/t10k-images.idx3-ubyte"
LABELS_PATH = "MNIST/t10k-labels.idx1-ubyte"

BAUD_RATE = 115200
TIMEOUT = 5.0
MAX_TEST_IMAGES = 100 # Change to 10000 for full test set

def find_stm32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in (p.manufacturer or ""):
            return p.device
    return None

def read_idx(filename):
    """Parses MNIST IDX files into numpy arrays."""
    with open(filename, 'rb') as f:
        # Read magic number and metadata
        magic, size = struct.unpack(">II", f.read(8))
        if magic == 2051: # Images
            rows, cols = struct.unpack(">II", f.read(8))
            print(f"Reading {size} images ({rows}x{cols})...")
            data = np.fromfile(f, dtype=np.uint8).reshape(size, rows * cols)
        elif magic == 2049: # Labels
            print(f"Reading {size} labels...")
            data = np.fromfile(f, dtype=np.uint8)
        else:
            raise ValueError("Invalid magic number in IDX file")
    return data

def main():
    # 1. Load Data
    try:
        images = read_idx(IMAGES_PATH)
        labels = read_idx(LABELS_PATH)
    except FileNotFoundError:
        print("❌ Error: IDX files not found in MNIST/ folder.")
        return

    # 2. Setup Serial
    port = find_stm32_port()
    if not port:
        print("❌ Error: STM32 Board not found!")
        return
    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)

    print(f"🚀 Starting Test on {MAX_TEST_IMAGES} images...")
    print("-" * 45)
    print(f"{'IDX':<5} | {'REAL':<5} | {'PRED':<5} | {'STATUS'}")
    print("-" * 45)

    correct = 0
    total = 0

    for i in range(MAX_TEST_IMAGES):
        image_bytes = images[i].tobytes() # Exactly 784 bytes
        real_digit = int(labels[i])

        # --- Handshake: Wait for [READY] ---
        ser.reset_input_buffer()
        ready = False
        t0 = time.time()
        while time.time() - t0 < 3.0:
            line = ser.readline().decode(errors='ignore').strip()
            if "[READY]" in line:
                ready = True
                break
        
        if not ready:
            print(f"[{i}] Timeout - MCU not sending [READY]")
            break

        # --- Send 784 bytes ---
        ser.write(image_bytes)

        # --- Wait for PREDICTION ---
        prediction = None
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            line = ser.readline().decode(errors='ignore').strip()
            if "PREDICTION:" in line:
                try:
                    prediction = int(line.split(":")[1].strip())
                except:
                    prediction = -1
                break
        
        # --- Results ---
        total += 1
        status = "✅" if prediction == real_digit else "❌"
        if status == "✅": correct += 1
        
        print(f"{i:<5} | {real_digit:<5} | {str(prediction):<5} | {status}")

    ser.close()
    if total > 0:
        print("-" * 45)
        print(f"🎯 Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()