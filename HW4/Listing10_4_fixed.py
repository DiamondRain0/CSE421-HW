import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# FIX: Reshape the weights to (2, 1) to match (input_shape, units)
w = tf.keras.initializers.Constant([[0.5], [-0.5]])
b = tf.keras.initializers.Constant(0.)

l0 = tf.keras.layers.Dense(units=1, 
                         input_shape=[2], 
                         kernel_initializer=w, 
                         bias_initializer=b, 
                         activation='sigmoid')

model = tf.keras.Sequential([l0])

# Compile is required for most TF operations
model.compile(optimizer='adam', loss='binary_crossentropy')

# Generate grid for visualization
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x_grid, y_grid = np.meshgrid(x, y)

# Prepare input data
x_gr_ravel = x_grid.ravel()
y_gr_ravel = y_grid.ravel()
input_data = np.c_[x_gr_ravel, y_gr_ravel]

# Get predictions
Z = model.predict(input_data)

# Plot the model output
Z_reshaped = Z.reshape(x_grid.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
surf = ax.plot_surface(x_grid, y_grid, Z_reshaped, cmap=cm.coolwarm, antialiased=True)

ax.set_xlabel('Input X')
ax.set_ylabel('Input Y')
ax.set_zlabel('Sigmoid Output')
ax.set_title('Visualization of a Single Neuron Output')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()