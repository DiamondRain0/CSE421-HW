import serial
import serial.tools.list_ports
import time
import csv
import struct
import sys
import numpy as np
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "./Part-1/NEW-DATA-1.T15.txt"
BAUD_RATE = 115200
TIMEOUT_SEC = 2.0         # Seconds to wait for MCU response
TEST_SAMPLES = 100        # How many samples to test
OUTPUT_CSV = "regression_results.csv"

# Protocol Signals (Must match C code)
SIGNAL_INIT      = "[INITIALIZED]"
SIGNAL_CONFIG_OK = "[CONFIG_OK]"
SIGNAL_READY     = "[READY]"
SIGNAL_SAMPLE_OK = "[SAMPLE_OK]"
SIGNAL_PREDICT   = "PREDICTED_VALUE:"
SIGNAL_END       = "[END_OF_PREDICTION]"

NUM_FEATURES = 5  # We used 5 lags (t-1 to t-5)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def find_stm_port():
    """Auto-detects ST-Link COM port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # Common descriptions for STM32 boards
        if "STMicroelectronics" in p.manufacturer or "ST-Link" in p.description:
            return p.device
    return None

def load_regression_data():
    """
    Loads data.txt and creates the same Lag Features (t-5...t-1)
    as the training script to ensure we test valid data.
    """
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        sys.exit(1)

    print("📥 Loading and processing Temperature Data...")
    
    columns = [
        "Date", "Time", "Temperature_Comedor_Sensor", "Temperature_Habitacion_Sensor",
        "Weather_Temperature", "CO2_Comedor_Sensor", "CO2_Habitacion_Sensor",
        "Humedad_Comedor_Sensor", "Humedad_Habitacion_Sensor", "Lighting_Comedor_Sensor",
        "Lighting_Habitacion_Sensor", "Precipitacion", "Meteo_Exterior_Crepusculo",
        "Meteo_Exterior_Viento", "Meteo_Exterior_Sol_Oest", "Meteo_Exterior_Sol_Est",
        "Meteo_Exterior_Sol_Sud", "Meteo_Exterior_Piranometro", "Exterior_Entalpic_1",
        "Exterior_Entalpic_2", "Exterior_Entalpic_turbo", "Temperature_Exterior_Sensor",
        "Humedad_Exterior_Sensor", "Day_Of_Week"
    ]

    df = pd.read_csv(DATA_PATH, sep=r"\s+", comment="#", names=columns, engine="python")
    
    # Target: Room Temperature (Downsampled)
    target_col = "Temperature_Habitacion_Sensor"
    y_raw = df[target_col][::4]
    
    # Create Lag Features (t-5 to t-1)
    X_df = pd.DataFrame()
    for i in range(NUM_FEATURES, 0, -1):
        X_df[f"t-{i}"] = y_raw.shift(i)

    # Clean NaNs
    X = X_df[NUM_FEATURES:].to_numpy(dtype=np.float32)
    y = y_raw[NUM_FEATURES:].to_numpy(dtype=np.float32)
    
    # Use the LAST 'TEST_SAMPLES' for testing
    X_test = X[-TEST_SAMPLES:]
    y_test = y[-TEST_SAMPLES:]
    
    print(f"✅ Loaded {len(X_test)} samples for testing.")
    return X_test, y_test

def wait_for_signal(ser, signal, timeout=TIMEOUT_SEC):
    """Reads serial lines until specific signal found."""
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if signal in line:
                    return True, line
                # print(f"DEBUG: {line}") # Uncomment to debug raw MCU output
            except: pass
    return False, ""

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Setup Serial
    port_name = find_stm_port()
    if not port_name:
        print("❌ Error: STM32 Board not found. Plug it in.")
        # Fallback for manual override if needed
        # port_name = "COM3" 
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
    X_test, y_test = load_regression_data()

    # 3. Handshake Phase (Send Configuration)
    print("🤝 Performing Handshake...")
    # Wait for MCU to say [INITIALIZED]
    # Note: If MCU is already running, you might need to press RESET on board
    found, _ = wait_for_signal(ser, SIGNAL_INIT, timeout=5.0)
    if not found:
        print("⚠️  MCU did not send [INITIALIZED]. Try pressing RESET on the board.")
        # We try sending config anyway in case it's stuck in loop
    
    # Send Number of Features (Integer 5) as 4 bytes
    print(f"📤 Sending Configuration: {NUM_FEATURES} features")
    ser.write(struct.pack('<I', NUM_FEATURES)) # <I = Little Endian Unsigned Int
    
    found, _ = wait_for_signal(ser, SIGNAL_CONFIG_OK)
    if not found:
        print("❌ Handshake Failed: MCU did not acknowledge config.")
        ser.close()
        sys.exit(1)
    print("✅ Handshake Complete.")

    # 4. Open CSV for Results
    try:
        csv_file = open(OUTPUT_CSV, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['Index', 'Actual_Temp', 'Predicted_Temp', 'Abs_Error'])
    except Exception as e:
        print(f"❌ Error opening CSV: {e}")
        ser.close()
        sys.exit(1)

    print("\n🚀 Starting Inference Loop...")
    print(f"{'IDX':<5} | {'ACTUAL':<8} | {'PREDICT':<8} | {'ERROR':<8}")
    print("-" * 40)

    total_mae = 0.0
    
    # 5. Testing Loop
    for i in range(TEST_SAMPLES):
        actual_temp = y_test[i]
        features = X_test[i] # Array of 5 floats
        
        # --- A. WAIT FOR [READY] ---
        found, _ = wait_for_signal(ser, SIGNAL_READY)
        if not found:
            print(f"⚠️ Timeout waiting for [READY] at idx {i}.")
            continue 

        # --- B. SEND 5 FLOATS ONE BY ONE ---
        # The C code receives 1 float, prints [SAMPLE_OK], receives next...
        for val in features:
            # Pack float (4 bytes little endian)
            ser.write(struct.pack('<f', val))
            
            # Wait for confirmation
            ack, _ = wait_for_signal(ser, SIGNAL_SAMPLE_OK, timeout=0.5)
            if not ack:
                print(f"⚠️ Sync Lost: No [SAMPLE_OK] for value {val}")
                break
        
        # --- C. GET PREDICTION ---
        found, line = wait_for_signal(ser, SIGNAL_PREDICT)
        pred_val = 0.0
        
        if found:
            try:
                # Parse "PREDICTED_VALUE: 24.1234"
                parts = line.split(':')
                pred_val = float(parts[1].strip())
            except ValueError:
                print(f"❌ Parse Error: {line}")
        
        # Consume the [END_OF_PREDICTION] marker
        wait_for_signal(ser, SIGNAL_END)

        # --- D. LOGGING ---
        abs_error = abs(actual_temp - pred_val)
        total_mae += abs_error
        
        print(f"{i:<5} | {actual_temp:<8.4f} | {pred_val:<8.4f} | {abs_error:<8.4f}")
        writer.writerow([i, actual_temp, pred_val, abs_error])

    # 6. Final Summary
    ser.close()
    csv_file.close()
    
    final_mae = total_mae / TEST_SAMPLES
    print("-" * 40)
    print(f"🎉 Test Complete!")
    print(f"📊 Mean Absolute Error (MAE): {final_mae:.4f}")
    print(f"💾 Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()