import serial
import serial.tools.list_ports
import time
import csv
import numpy as np
from sklearn import datasets
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BAUD_RATE = 115200
TIMEOUT_SEC = 5.0       # Wait up to 5s for MCU to process
TEST_SAMPLES = 500      # Number of images to test
OUTPUT_CSV = "svm_results.csv"

# Signals expected from MCU
SIGNAL_READY = "[READY]"
SIGNAL_PREDICT = "PREDICTED_CLASS:"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def find_stm_port():
    """Auto-detects ST-Link COM port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in p.manufacturer or "ST-Link" in p.description:
            return p.device
    return None

def load_mnist_test_data():
    """Fetches MNIST and returns the last N samples."""
    print("📥 Loading MNIST dataset (this takes 10-30 seconds)...")
    # Fetch data (cache=True helps on 2nd run)
    mnist = datasets.fetch_openml('mnist_784', version=1, parser='auto', cache=True)
    
    X = mnist.data.astype(np.uint8).to_numpy()
    y = mnist.target.astype(int).to_numpy()
    
    # Take the LAST 'TEST_SAMPLES' images
    X_test = X[-TEST_SAMPLES:]
    y_test = y[-TEST_SAMPLES:]
    
    print(f"✅ Loaded {len(X_test)} test images.")
    return X_test, y_test

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup Serial
    port_name = find_stm_port()
    if not port_name:
        print("❌ Error: STM32 Board not found. Plug it in.")
        sys.exit(1)
        
    print(f"🔌 Connecting to {port_name} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=0.1)
        time.sleep(2) # Wait for board reset
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        sys.exit(1)

    # 2. Load Data
    X_test, y_test = load_mnist_test_data()

    # 3. Open CSV for Results
    try:
        csv_file = open(OUTPUT_CSV, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['Image_Index', 'Real_Label', 'Predicted_Label', 'Is_Correct'])
    except Exception as e:
        print(f"❌ Error opening CSV: {e}")
        ser.close()
        sys.exit(1)

    print("\n🚀 Starting Inference Loop...")
    print(f"{'IDX':<5} | {'REAL':<5} | {'PRED':<5} | {'RESULT':<10} | {'ACCURACY':<10}")
    print("-" * 55)

    correct_count = 0
    
    # 4. Testing Loop
    for i in range(TEST_SAMPLES):
        real_label = y_test[i]
        
        # Get raw bytes (784 bytes)
        # Your C code expects 0-255 values and thresholds them on board.
        image_bytes = X_test[i].tobytes()
        
        # --- A. SYNC: WAIT FOR [READY] ---
        # We must wait for the MCU to print [READY] before sending data
        ready_received = False
        start_wait = time.time()
        
        while time.time() - start_wait < TIMEOUT_SEC:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    if SIGNAL_READY in line:
                        ready_received = True
                        break
                except: pass
        
        if not ready_received:
            print(f"⚠️ Timeout waiting for [READY] at idx {i}. Retrying...")
            continue 

        # --- B. SEND IMAGE DATA ---
        ser.write(image_bytes)
        
        # --- C. WAIT FOR PREDICTION ---
        predicted_label = -1
        start_wait = time.time()
        
        while time.time() - start_wait < TIMEOUT_SEC:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    # Look for "PREDICTED_CLASS: 5"
                    if SIGNAL_PREDICT in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            predicted_label = int(parts[1].strip())
                            break
                    # Optional: Print other debug lines from MCU
                    # else: print(f"STM32: {line}") 
                except: pass
        
        # --- D. PROCESS RESULT ---
        if predicted_label != -1:
            is_correct = (predicted_label == real_label)
            if is_correct: correct_count += 1
            
            icon = "✅" if is_correct else "❌"
            current_acc = (correct_count / (i + 1)) * 100
            
            print(f"{i:<5} | {real_label:<5} | {predicted_label:<5} | {icon:<10} | {current_acc:.1f}%")
            
            writer.writerow([i, real_label, predicted_label, 1 if is_correct else 0])
        else:
            print(f"{i:<5} | {real_label:<5} | {'TIMEOUT'} | {'❌'}         | N/A")
            writer.writerow([i, real_label, -1, 0])

    # 5. Final Summary
    ser.close()
    csv_file.close()
    
    final_acc = (correct_count / TEST_SAMPLES) * 100
    print("-" * 55)
    print(f"🎉 Test Complete!")
    print(f"📊 Final Accuracy: {final_acc:.2f}%")
    print(f"💾 Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()