import serial
import serial.tools.list_ports
import struct
import time
import csv
import os
import glob
import numpy as np
import librosa
import random

# =========================
# CONFIGURATION
# =========================
TEST_DATA_DIR = "recordings"      # Folder with FSDD .wav files
BAUD_RATE = 115200
SAMPLE_RATE = 8000
NUM_TEST_SAMPLES = 300             # Number of random files to test
OUTPUT_CSV = "knn_random_test_results.csv"

# --- TIMEOUTS (Increased) ---
TIMEOUT_READY = 10.0              # Time to wait for [READY] signal
TIMEOUT_PREDICT = 5.0             # Time to wait for prediction result

# =========================
# HELPER FUNCTIONS
# =========================
def find_stm_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in p.manufacturer or "ST-Link" in p.description:
            return p.device
    return None

def wait_for_ready(ser, timeout=TIMEOUT_READY):
    """Waits for the board to send [READY]"""
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if "[READY]" in line: return True
            except: pass
    return False

def get_prediction(ser, timeout=TIMEOUT_PREDICT):
    """Waits for PREDICTED_CLASS: X"""
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if "PREDICTED_CLASS:" in line:
                    return line.split(":")[1].strip()
            except: pass
    return "TIMEOUT"

# =========================
# MAIN EXECUTION
# =========================
def main():
    # 1. Load File List
    all_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.wav"))
    if not all_files:
        print(f"❌ No .wav files found in {TEST_DATA_DIR}")
        return

    # 2. Select Random Samples
    if len(all_files) > NUM_TEST_SAMPLES:
        test_files = random.sample(all_files, NUM_TEST_SAMPLES)
    else:
        test_files = all_files # Use all if less than requested
        random.shuffle(test_files)

    print(f"🎲 Selected {len(test_files)} random files for testing.")

    # 3. Connect to Board
    port = find_stm_port()
    if not port:
        print("❌ STM32 Board not found. Plug it in!")
        return
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"🔌 Connected to {port}. Resetting board...")
        time.sleep(2) 
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        return

    # 4. Wait for Initialization
    print("⏳ Waiting for board initialization...")
    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if "[INITIALIZED]" in line: break
            if "[READY]" in line: break 
        except: pass

    print("✅ Board Ready. Starting Test Loop...")
    
    # 5. Open CSV for writing
    try:
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write Header
            writer.writerow(['Filename', 'Real_Label', 'Predicted_Label', 'Is_Correct'])

            print("-" * 75)
            print(f"{'FILENAME':<25} | {'REAL':<5} | {'PRED':<8} | {'RESULT'}")
            print("-" * 75)

            correct_count = 0
            total_count = 0

            # 6. Test Loop
            for f_path in test_files:
                filename = os.path.basename(f_path)
                
                # Extract Real Label (e.g. "0_jackson_5.wav" -> "0")
                real_label = filename.split('_')[0]

                # Process Audio (Float -> Int16)
                try:
                    audio, _ = librosa.load(f_path, sr=SAMPLE_RATE)
                    audio, _ = librosa.effects.trim(audio)
                    audio_int16 = (audio * 32767).astype(np.int16)
                except:
                    print(f"⚠️ Error loading {filename}, skipping.")
                    continue
                
                if len(audio_int16) < 1024: continue # Too short

                # Handshake with Timeout Retry
                if not wait_for_ready(ser):
                    print("⚠️ Board not ready. Resyncing...")
                    ser.write(b'\n')
                    time.sleep(0.2)
                    ser.reset_input_buffer()
                    if not wait_for_ready(ser): 
                        print("❌ Sync failed. Skipping.")
                        continue

                # Send Size (4 bytes)
                ser.write(struct.pack('<I', len(audio_int16)))
                
                # Send Data (PCM Int16)
                ser.write(audio_int16.tobytes())

                # Get Result
                pred_label = get_prediction(ser)

                # Compare
                is_correct = (pred_label == real_label)
                if is_correct: correct_count += 1
                total_count += 1

                # Print to Console
                icon = "✅" if is_correct else "❌"
                print(f"{filename:<25} | {real_label:<5} | {pred_label:<8} | {icon}")

                # Save to CSV
                writer.writerow([filename, real_label, pred_label, '1' if is_correct else '0'])
                
                # Small delay to let UART settle
                time.sleep(0.05)

            # 7. Final Stats
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            print("-" * 75)
            print(f"📊 Final Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
            print(f"💾 Detailed results saved to: {OUTPUT_CSV}")
            print("-" * 75)

    except IOError as e:
        print(f"❌ File I/O Error (Close the CSV if it's open!): {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()