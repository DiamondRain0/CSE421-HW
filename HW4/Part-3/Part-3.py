import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def extract_hu_moments(images):
    hu_features = []
    for img in images:
        # 1. Moments calculation
        # MNIST images are 28x28. cv2.moments calculates spatial/central moments.
        moments = cv2.moments(img)
        
        # 2. Hu Moments (7 values that are invariant to scale, rotation, and translation)
        hu = cv2.HuMoments(moments).flatten()
        
        # 3. Log Transform
        # Hu moments have a massive dynamic range (e.g., 10^-3 to 10^-20).
        # We use a log transform to make them manageable for the Neural Network.
        hu = -1 * np.sign(hu) * np.log10(np.abs(hu) + 1e-15)
        
        hu_features.append(hu)
    return np.array(hu_features)

# --- 1. Load MNIST Dataset from Keras ---
print("Loading MNIST from Keras...")
(x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()

# --- 2. Feature Extraction (Hu Moments) ---
print("Extracting Hu Moments (this may take a few seconds)...")
# We process the images to get the 7 features required by Listing 11.6
X_train_hu = extract_hu_moments(x_train_raw)
X_test_hu = extract_hu_moments(x_test_raw)

# --- 3. Normalization ---
# Neural networks perform best when features have a mean of 0 and std of 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_hu)
X_test = scaler.transform(X_test_hu)

# --- 4. Build Model (Architecture from Listing 11.6) ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(7,)), # 7 Hu Moments
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax") # 10 digits (0-9)
])

# --- 5. Compile and Train ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks as mentioned in Listing 11.6
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
    tf.keras.callbacks.ModelCheckpoint("mlp_mnist_model.h5", save_best_only=True)
]

print("Starting training...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# --- 6. Evaluation ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Save the final model
model.save("mlp_mnist_hu_final.h5")