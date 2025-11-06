import serial
import serial.tools.list_ports
import threading
import time
import os
import wave
import struct
from queue import Queue

# =========================
# USER CONFIGURATION
# =========================
FOLDER_PATH = r"data"   # 📁 folder with WAV files
BAUD_RATE = 115200
WAIT_BETWEEN_FILES = 2             # seconds between files
RETRY_LIMIT = 3                    # max retry attempts
CMD_SEND_AUDIO = b'\x01'
END_MARKER = "[END_OF_FEATURES]"
CHUNK_SIZE = 256
OUTPUT_LOG = "mfcc_results.txt"

# Thread-safe message queue for Mbed output
mbed_queue = Queue()

# =========================
# HELPER FUNCTIONS
# =========================
def find_mbed_port():
    """Detect Mbed COM port automatically."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if "mbed" in desc or "usb serial device" in desc or "stm" in desc:
            print(f"[INFO] Mbed board found on {p.device}")
            return p.device
    raise RuntimeError("❌ Mbed board not found! Plug it in and try again.")


def serial_reader(ser):
    """Continuously read Mbed output lines and push into queue."""
    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"[MBED] {line}")
                mbed_queue.put(line)
        except serial.SerialException:
            break
        except Exception:
            continue


def send_audio_file(ser, wav_path, log_file):
    """Send one WAV file to Mbed and wait for MFCC results."""
    print(f"\n🎵 Sending: {os.path.basename(wav_path)}")

    # Read the WAV file
    with wave.open(wav_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        sampwidth = wav_file.getsampwidth()

        if sampwidth != 2:
            raise ValueError("WAV must be 16-bit PCM format.")

        raw_data = wav_file.readframes(num_frames)
        samples = struct.unpack("<" + "h" * num_frames * num_channels, raw_data)

        if num_channels > 1:
            samples = samples[::num_channels]

    input_size_bytes = len(samples) * 2
    print(f"[INFO] {len(samples)} samples @ {sample_rate} Hz ({input_size_bytes} bytes)")

    # Send header
    ser.write(CMD_SEND_AUDIO)
    ser.write(struct.pack("<I", len(samples)))
    ser.write(struct.pack("<I", sample_rate))
    ser.flush()

    # Send audio data in chunks
    for i in range(0, len(samples), CHUNK_SIZE):
        chunk = samples[i:i + CHUNK_SIZE]
        ser.write(struct.pack("<" + "h" * len(chunk), *chunk))
        ser.flush()
        time.sleep(0.002)

    print(f"[INFO] Audio file sent ({input_size_bytes} bytes). Waiting for MFCC results...")

    # Wait for END_MARKER
    mfcc_lines = []
    start_time = time.time()
    timeout = 20  # seconds per file

    while True:
        try:
            line = mbed_queue.get(timeout=1)
            mfcc_lines.append(line)

            if END_MARKER in line:
                print("✅ MFCC computation complete.")
                break

        except Exception:
            if time.time() - start_time > timeout:
                print("⚠️ Timeout waiting for Mbed response.")
                return None

    # Extract coefficient lines
    coeffs = [l for l in mfcc_lines if "Coeff" in l]
    output_data = "\n".join(coeffs).encode('utf-8')
    output_size_bytes = len(output_data)

    # Print and log
    print(f"\n📊 --- MFCC RESULT ---")
    if coeffs:
        for c in coeffs:
            print("  " + c.replace("[MBED]", "").strip())
    else:
        print("  (No coefficients printed)")
    print("--------------------------")
    print(f"💾 {input_size_bytes} bytes audio  --->  {output_size_bytes} bytes MFCC features\n")

    # Save to log file
    with open(log_file, "a") as f:
        f.write(f"File: {os.path.basename(wav_path)}\n")
        f.write(f"Audio size: {input_size_bytes} bytes\n")
        f.write(f"Feature size: {output_size_bytes} bytes\n")
        if coeffs:
            for c in coeffs:
                f.write(c.replace("[MBED]", "").strip() + "\n")
        f.write("-" * 40 + "\n")

    return coeffs


# =========================
# MAIN
# =========================
def main():
    print("\n🔍 Scanning for Mbed board...")
    SERIAL_PORT = find_mbed_port()

    print(f"[INFO] Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
    time.sleep(2)

    threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()

    # Load all .wav files
    wav_files = [os.path.join(FOLDER_PATH, f)
                 for f in os.listdir(FOLDER_PATH)
                 if f.lower().endswith(".wav")]
    wav_files.sort()

    if not wav_files:
        print("❌ No WAV files found.")
        return

    print(f"[INFO] Found {len(wav_files)} files.\n")

    # Clear or create log file
    with open(OUTPUT_LOG, "w") as f:
        f.write("=== MFCC Extraction Log ===\n\n")

    failed = []

    for idx, wav_path in enumerate(wav_files, 1):
        print(f"[{idx}/{len(wav_files)}]")
        success = False

        for attempt in range(1, RETRY_LIMIT + 1):
            print(f"Attempt {attempt}/{RETRY_LIMIT}")
            coeffs = send_audio_file(ser, wav_path, OUTPUT_LOG)

            if coeffs is not None:
                success = True
                break
            else:
                print(f"⚠️ Retrying {os.path.basename(wav_path)} ...")
                time.sleep(2)
                ser.reset_input_buffer()
                ser.reset_output_buffer()

        if not success:
            print(f"❌ Failed after {RETRY_LIMIT} attempts: {os.path.basename(wav_path)}")
            failed.append(wav_path)

        if idx < len(wav_files):
            print(f"⏳ Waiting {WAIT_BETWEEN_FILES}s before next file...")
            time.sleep(WAIT_BETWEEN_FILES)

    print("\n✅ Processing complete.")
    if failed:
        print("\n⚠️ The following files failed:")
        for f in failed:
            print("  -", os.path.basename(f))

    ser.close()

    print(f"\n🗂️ Results saved to: {OUTPUT_LOG}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
