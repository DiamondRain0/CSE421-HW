import serial
import serial.tools.list_ports
import struct
import time
import csv
import pandas as pd
from scipy import stats

# =========================
# USER CONFIGURATION
# =========================
WISDM_DATA_FILE = "WISDM/test.csv" 
WINDOW_SIZE = 128
MAX_WINDOWS_TO_PROCESS = 3000 # Set limit for testing
BAUD_RATE = 115200
OUTPUT_CSV_FILE = "bayes_accuracy_results.csv"

# --- Protocol markers ---
INITIALIZED_MARKER = "[INITIALIZED]"
CONFIG_OK_MARKER = "[CONFIG_OK]"
READY_MARKER = "[READY]"
SAMPLE_OK_MARKER = "[SAMPLE_OK]"
PROCESS_MARKER = "[PROCESS]\n"
END_MARKER = "[END_OF_PREDICTION]" # Matches C code

ACTIVITY_MAP = {
    0: "Downstairs", 1: "Jogging", 2: "Sitting",
    3: "Standing",   4: "Upstairs",  5: "Walking"
}

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Initial load of {len(df)} rows from {filepath}")
        column_names = ['acc_x', 'acc_y', 'acc_z', 'timestamp', 'activity']
        df.columns = column_names[:len(df.columns)]
        for col in ['acc_x', 'acc_y', 'acc_z', 'activity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(how='any', inplace=True)
        df['activity'] = df['activity'].astype(int)
        print(f"[INFO] Successfully loaded and cleaned {len(df)} data points.")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def find_stm_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "STMicroelectronics" in p.manufacturer or "ST-Link" in p.description:
            return p.device
    return None

def wait_for_signal(ser, signal, timeout=5, capture_lines=None):
    start_time = time.time()
    buffer = ""
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            try:
                char = ser.read().decode(errors='ignore')
                buffer += char
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    # print(f"[STM32] {line}") # Uncomment to see everything
                    if capture_lines is not None: capture_lines.append(line)
                    if signal in line: return True
            except Exception: pass
    return False

# =========================
# MAIN
# =========================
def main():
    wisdm_df = load_and_preprocess_data(WISDM_DATA_FILE)
    if wisdm_df is None: return

    com_port = find_stm_port()
    if not com_port: print("❌ STM32 not found"); return
    
    try:
        ser = serial.Serial(com_port, BAUD_RATE, timeout=1)
        print(f"[INFO] Connected to {com_port}")
        time.sleep(2); ser.reset_input_buffer()
    except Exception as e: print(e); return

    # --- CONFIGURATION ---
    print("[INFO] Waiting for STM32 initialization...")
    if not wait_for_signal(ser, INITIALIZED_MARKER, timeout=10):
        print("❌ STM32 did not initialize."); ser.close(); return
    
    print(f"[INFO] Sending Window Size: {WINDOW_SIZE}")
    ser.write(struct.pack('<I', WINDOW_SIZE)); ser.flush()

    if not wait_for_signal(ser, CONFIG_OK_MARKER):
        print("❌ Config failed."); ser.close(); return

    # --- CSV SETUP ---
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # New Header: No features, just results
            header = ['Window_Index', 'Real_Label', 'Real_Activity', 
                      'Predicted_Label', 'Predicted_Activity', 'Is_Correct']
            writer.writerow(header)
        print(f"💾 Created {OUTPUT_CSV_FILE}")
    except IOError as e: print(e); ser.close(); return

    label_buffer, samples_in_window, window_count = [], 0, 0
    correct_count = 0

    if not wait_for_signal(ser, READY_MARKER):
        print("❌ No READY signal."); ser.close(); return

    print("[INFO] Starting Classification Loop...")

    for index, row in wisdm_df.iterrows():
        # Send Data
        ser.write(struct.pack('<fff', row['acc_x'], row['acc_y'], row['acc_z']))
        
        # Wait for ACK (Fast timeout)
        if not wait_for_signal(ser, SAMPLE_OK_MARKER, timeout=0.5):
            print(f"❌ Missed ACK at sample {samples_in_window}"); break
        
        label_buffer.append(row['activity'])
        samples_in_window += 1

        # Window Full?
        if samples_in_window == WINDOW_SIZE:
            # 1. Determine Real Label (Mode of buffer)
            real_label = int(stats.mode(label_buffer, keepdims=False)[0])
            real_name = ACTIVITY_MAP.get(real_label, "Unknown")

            # 2. Trigger STM32 Processing
            ser.write(PROCESS_MARKER.encode()); ser.flush()

            # 3. Capture Output
            lines = []
            if wait_for_signal(ser, END_MARKER, timeout=2.0, capture_lines=lines):
                predicted_label = -1
                
                # Parse the specific line "PREDICTED_CLASS: X"
                for line in lines:
                    if "PREDICTED_CLASS:" in line:
                        try:
                            parts = line.split(':')
                            predicted_label = int(parts[1].strip())
                        except: pass
                
                pred_name = ACTIVITY_MAP.get(predicted_label, "Unknown")
                
                # 4. Check Correctness
                is_correct = 1 if real_label == predicted_label else 0
                if is_correct: correct_count += 1
                
                # 5. Save to CSV
                with open(OUTPUT_CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([window_count, real_label, real_name, 
                                     predicted_label, pred_name, is_correct])
                
                print(f"Window {window_count}: Real={real_name} vs Pred={pred_name} [{'CORRECT' if is_correct else 'WRONG'}]")
            else:
                print(f"❌ Timeout waiting for prediction window {window_count}")
                break

            # Reset for next window
            window_count += 1
            samples_in_window = 0
            label_buffer.clear()
            
            if window_count >= MAX_WINDOWS_TO_PROCESS: break
            if not wait_for_signal(ser, READY_MARKER): break

    accuracy = (correct_count / window_count) * 100 if window_count > 0 else 0
    print("\n" + "="*60)
    print(f"✅ Finished. Processed {window_count} windows.")
    print(f"📊 Accuracy: {accuracy:.2f}%")
    print(f"💾 Results saved to {OUTPUT_CSV_FILE}")
    ser.close()

if __name__ == "__main__":
    main()