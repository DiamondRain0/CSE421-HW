import os
import numpy as np
import scipy.io.wavfile as wav
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

def extract_improved_features(file_list):
    feats, labels = [], []
    # Target length for 1 second of audio at 8000Hz
    target_len = 8000 
    
    for f in file_list:
        try:
            sample_rate, sig = wav.read(f)
            # Ensure signal is mono
            if len(sig.shape) > 1: sig = sig[:, 0]
            
            # 1. Normalize Length: Pad or Truncate to 1 second
            if len(sig) < target_len:
                sig = np.pad(sig, (0, target_len - len(sig)), 'constant')
            else:
                sig = sig[:target_len]
            
            # 2. Get FFT and Log-Scale (Decibel-like scale)
            spec = np.abs(np.fft.rfft(sig, n=1024))
            # Split into 26 bins as per Listing 11.5
            feat = [np.mean(x) for x in np.array_split(spec, 26)]
            
            # 3. Apply Log transform (essential for audio)
            feat = np.log10(np.array(feat) + 1e-6)
            
            feats.append(feat)
            labels.append(os.path.basename(f).split('_')[0])
        except Exception as e:
            continue
            
    return np.array(feats), np.array(labels)

# --- Main Execution ---
AUDIO_DIR = "recordings"

if os.path.exists(AUDIO_DIR):
    all_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    train_files = [f for f in all_files if "yweweler" not in f]
    
    X_train_raw, y_train = extract_improved_features(train_files)
    
    # 4. Standardize: Scaled data (Mean=0, Std=1) makes NNs work
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    ohe = OneHotEncoder(sparse_output=False)
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
    num_classes = len(ohe.categories_[0])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(26,)),
        tf.keras.layers.Dense(128, activation="relu"), # Increased slightly for capacity
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # Using a slightly lower learning rate for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])

    print(f"Starting training on {num_classes} classes...")
    history = model.fit(X_train, y_train_ohe, epochs=100, batch_size=16, verbose=1)

    model.save("mlp_kws_model.h5")
    print("\nFinal Accuracy: ", history.history['accuracy'][-1])
else:
    print(f"Error: Recordings folder '{AUDIO_DIR}' not found.")