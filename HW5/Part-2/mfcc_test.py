import serial
import serial.tools.list_ports
import struct
import time
import os
import glob
import numpy as np
import librosa
import random

# --- CONFIG ---
TEST_DATA_DIR = "recordings/" 
BAUD_RATE = 115200
TIMEOUT = 10.0

def get_audio_data(file_path):
    """Load and fix audio to exactly 16,000 samples (int16)"""
    try:
        y, _ = librosa.load(file_path, sr=16000, duration=1.0)
        y = librosa.util.fix_length(y, size=16000)
        return (y * 32767).astype(np.int16)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in (p.manufacturer or ""): return p.device
    return None

def main():
    port = find_port()
    if not port:
        print("❌ STM32 not found!")
        return

    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    ser.reset_input_buffer()
    
    files = glob.glob(os.path.join(TEST_DATA_DIR, "*.wav"))
    if not files:
        print(f"❌ No files found in {TEST_DATA_DIR}")
        return
        
    random.shuffle(files)
    print(f"🔌 Connected to {port}. Testing {len(files)} files...")

    for i, fpath in enumerate(files[:50]):
        filename = os.path.basename(fpath)
        real_digit = filename.split('_')[0]
        audio = get_audio_data(fpath)
        
        print(f"\n[{i+1}] Processing: {filename} (Expect: {real_digit})")
        
        # --- 1. Wait for [READY] ---
        ready = False
        t0 = time.time()
        while time.time() - t0 < 5.0:
            line = ser.readline().decode(errors='ignore').strip()
            if line: print(f"  MCU: {line}")
            if "[READY]" in line:
                ready = True
                break
        
        if not ready:
            print("  ❌ MCU never sent [READY]")
            continue

        # --- 2. Send Audio (32,000 bytes) ---
        print("  >> Sending Audio...")
        ser.write(audio.tobytes())

        # --- 3. Wait for Prediction ---
        prediction = None
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            line = ser.readline().decode(errors='ignore').strip()
            if line: print(f"  MCU: {line}")
            if "PREDICTION:" in line:
                prediction = line.split(':')[1].strip()
                break
        
        if prediction is not None:
            status = "✅ MATCH" if prediction == real_digit else "❌ WRONG"
            print(f"  >> RESULT: Predicted {prediction} | {status}")
        else:
            print("  ❌ No prediction received")

    ser.close()
    print("\nDone.")

if __name__ == "__main__":
    main()