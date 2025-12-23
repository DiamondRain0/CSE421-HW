import os.path as osp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# --- Data Loading Utility ---
def read_data(file_path):
    column_names = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    
    # We use on_bad_lines='skip' to ignore the 0.1% of lines that are malformed
    # We also use engine='python' which is more flexible for messy text files
    df = pd.read_csv(
        file_path, 
        header=None, 
        names=column_names, 
        on_bad_lines='skip', 
        engine='python'
    )
    
    # Clean the 'z' column: WISDM has a semicolon at the end of every line (e.g. "0.1234;")
    # This turns the string "0.1234;" into the float 0.1234
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    
    df.dropna(axis=0, how='any', inplace=True)
    return df

def create_features(df, time_steps, step):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x'].values[i: i + time_steps]
        ys = df['y'].values[i: i + time_steps]
        zs = df['z'].values[i: i + time_steps]
        # Extracting 10 features as required by Listing 11.4 input_shape=[10]
        # (Means, Std Devs, and Max for x,y,z + Mean of sums)
        features = [np.mean(xs), np.mean(ys), np.mean(zs), 
                    np.std(xs), np.std(ys), np.std(zs), 
                    np.max(xs), np.max(ys), np.max(zs),
                    np.mean(xs+ys+zs)]
        segments.append(features)
        labels.append(df['activity'].values[i])
    return np.array(segments), np.array(labels)

# --- Main Logic from Listing 11.4 ---
DATA_PATH = osp.join("WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
TIME_PERIODS = 80
STEP_DISTANCE = 40

if osp.exists(DATA_PATH):
    data_df = read_data(DATA_PATH)
    df_train = data_df[data_df["user"] <= 28]
    df_test = data_df[data_df["user"] > 28]

    train_segments_np, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
    test_segments_np, test_labels = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=[10], activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    ohe = OneHotEncoder()
    train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()
    categories = np.unique(test_labels)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])

    model.fit(train_segments_np, train_labels_ohe, epochs=50, verbose=1)

    nn_preds = model.predict(test_segments_np)
    predicted_classes = np.argmax(nn_preds, axis=1)
    
    # Label decoding for confusion matrix
    test_labels_int = ohe.transform(test_labels.reshape(-1,1)).toarray().argmax(axis=1)
    conf_matrix = confusion_matrix(test_labels_int, predicted_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=categories)
    cm_display.plot()
    plt.title("Neural Network Confusion Matrix")
    plt.show()
    model.save("mlp_har_model.h5")
else:
    print(f"File {DATA_PATH} not found.")