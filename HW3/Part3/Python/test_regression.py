import serial
import serial.tools.list_ports
import struct
import time
import numpy as np
import os
import glob
import librosa
import random
import csv # Import CSV library

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "recordings" # Path where your WAV files are stored
BAUD_RATE = 115200
TIMEOUT = 5.0 
INITIAL_WAIT_TIME = 5.0 
TEST_SAMPLES = 100 

# Output CSV file configuration
CSV_OUTPUT_FILE = "prediction_results.csv"

# Audio Parameters (MUST MATCH MCU CONFIG)
SAMPLE_RATE = 8000
AUDIO_SAMPLES = SAMPLE_RATE 
AUDIO_BUFFER_SIZE_BYTES = AUDIO_SAMPLES * 2 


# ==========================================
# DATA LOADING AND PREPARATION
# ==========================================

def load_and_prepare_audio(file_path):
    """ Loads audio and prepares it as a 16-bit array. """
    try:
        audio_float, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        
        if len(audio_int16) > AUDIO_SAMPLES:
            audio_int16 = audio_int16[:AUDIO_SAMPLES]
        elif len(audio_int16) < AUDIO_SAMPLES:
            padding = np.zeros(AUDIO_SAMPLES - len(audio_int16), dtype=np.int16)
            audio_int16 = np.concatenate([audio_int16, padding])
            
        return audio_int16

    except Exception:
        return None

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
    wav_files = glob.glob(os.path.join(DATASET_PATH, "*.wav"))
    
    if not wav_files:
        print("❌ No recordings found in 'recordings/' directory. Cannot run test.")
        return

    # 1. Prepare Test Data with Filtering
    print("Preparing audio buffers (16-bit, 8kHz, 1s window) with file filtering...")
    
    test_data = []
    
    for f in wav_files:
        try:
            filename = os.path.basename(f)
            label_prefix = filename.split('_')[0]
            
            # --- FILTERING LOGIC: ONLY USE FILES STARTING WITH '0' or '1' ---
            if label_prefix == '0':
                real_bin = 1 # Target Class
            elif label_prefix == '1':
                real_bin = 0 # Other Class
            else:
                continue # Skip all other files
            # -----------------------------------------------------------------

            audio_buffer = load_and_prepare_audio(f)
            if audio_buffer is not None:
                test_data.append((audio_buffer, real_bin, filename))
                
        except Exception:
            continue
    
    if not test_data:
        print("❌ Failed to load or prepare any audio files that started with '0' or '1'.")
        return

    random.shuffle(test_data)
    test_data = test_data[:min(len(test_data), TEST_SAMPLES)]
    print(f"Testing on {len(test_data)} filtered samples.")

    # 2. Connect to MCU
    port = find_port()
    if not port: 
        print("❌ STM32 Board not found. Is it connected and drivers installed?")
        return
    
    print(f"🔌 Connecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.5) 
        
        print(f"Waiting {INITIAL_WAIT_TIME}s for MCU boot and MFCC setup...")
        time.sleep(INITIAL_WAIT_TIME) 
        ser.reset_input_buffer()
        print("Input buffer cleared.")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 3. Test Loop and CSV Setup
    total_processed = 0
    correct_count = 0
    results_list = []
    
    print(f"\n🚀 Starting Controller Test...")
    print(f"Payload size: {AUDIO_BUFFER_SIZE_BYTES} bytes (8000 x int16_t)")
    print("-" * 50)
    print(f"{'#':<4} | {'INPUT CLASS':<11} | {'PRED':<5} | {'RES':<4} | {'ACCURACY':<8}")
    print("-" * 50)

    for i, (audio_buffer, real_bin, filename) in enumerate(test_data):
        
        # --- A. Handshake (Wait for Ready) ---
        ready = False
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    if "[READY]" in line: 
                        ready = True
                        break
                except: 
                    pass
        
        if not ready:
            result_type = "Timeout"
            pred = -1
            print(f"{i+1:<4} | {'--':<11} | {'--':<5} | {'--':<4} | ⚠️ Timeout waiting for Ready.")
            ser.reset_input_buffer()
        else:
            # --- B/C/D. Send, Read, and Calculate Stats ---
            
            # B. Send Data
            try:
                payload = struct.pack(f'<{AUDIO_SAMPLES}h', *audio_buffer)
                ser.write(payload)
            except Exception:
                result_type = "SendError"
                pred = -1
                
            # C. Read Prediction
            pred = -1
            prob_str = "N/A"
            t_read = time.time()
            while time.time() - t_read < TIMEOUT:
                if ser.in_waiting:
                    try:
                        line = ser.readline().decode(errors='ignore').strip()
                        if "PREDICTION:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                pred_raw = parts[1].split('(')
                                pred = int(pred_raw[0].strip())
                                prob_str = pred_raw[1].replace('Prob:', '').replace(')', '').strip()
                            break
                    except: pass
            
            # D. Stats
            if pred != -1:
                total_processed += 1
                is_correct = (pred == real_bin)
                
                if is_correct: 
                    correct_count += 1
                    icon = "✅"
                    result_type = "TP" if real_bin == 1 else "TN"
                else:
                    icon = "❌"
                    result_type = "FN" if real_bin == 1 else "FP"

                current_acc = (correct_count / total_processed) * 100.0
                class_str = "Target (1)" if real_bin == 1 else "Other (0)"
                
                print(f"{i+1:<4} | {class_str:<11} | {pred:<5} | {icon:<4} | {current_acc:.1f}%")
            else:
                result_type = "ReadError"
                print(f"{i+1:<4} | {'--':<11} | {'--':<5} | {'--':<4} | Error/Timeout reading prediction.")
        
        # E. Record Result for CSV
        results_list.append({
            'Index': i + 1,
            'Filename': filename,
            'Actual_Class': real_bin,
            'Predicted_Class': pred if pred != -1 else 'N/A',
            'Result_Type': result_type,
            'Correct': is_correct if pred != -1 else 'N/A'
        })
        
    # 4. Final Result Summary and CSV Write
    print("-" * 50)
    if total_processed > 0:
        final_acc = (correct_count / total_processed) * 100.0
        print(f"🎉 DONE. Final Accuracy: {final_acc:.2f}% (Total tested: {total_processed})")
        
        tp = sum(1 for r in results_list if r['Result_Type'] == 'TP')
        tn = sum(1 for r in results_list if r['Result_Type'] == 'TN')
        fp = sum(1 for r in results_list if r['Result_Type'] == 'FP')
        fn = sum(1 for r in results_list if r['Result_Type'] == 'FN')
        
        print(f"Confusion Matrix (Total Processed: {total_processed}):")
        print(f"  True Positives (TP - Target Correct): {tp}")
        print(f"  True Negatives (TN - Other Correct): {tn}")
        print(f"  False Positives (FP - Other Wrong): {fp}")
        print(f"  False Negatives (FN - Target Missed): {fn}")
        
    else:
        print("Test completed but no predictions were successfully received.")
        
    ser.close()

    # Write results to CSV
    with open(CSV_OUTPUT_FILE, 'w', newline='') as f:
        fieldnames = ['Index', 'Filename', 'Actual_Class', 'Predicted_Class', 'Probability', 'Result_Type', 'Correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results_list)
        
    print(f"\n✅ Detailed results saved to {CSV_OUTPUT_FILE}")

if __name__ == "__main__":
    main()