
# 6. Model-Based Anomaly Detection: Autoencoders


## Concept & Theory


Autoencoders are a type of neural network used for unsupervised learning, particularly for dimensionality reduction and feature learning. For anomaly detection, the idea is to train an autoencoder on normal data. The autoencoder learns to reconstruct the normal data with low error. When the autoencoder is presented with an anomalous data point, it will have a high reconstruction error. This reconstruction error can be used as an anomaly score.


## Mathematical Overview


1. **Encoder**: The encoder is a neural network that maps the input data *x* to a lower-dimensional representation *z*, called the bottleneck or latent space.   $ z = f(x) $
2. **Decoder**: The decoder is a neural network that reconstructs the input data from the latent representation *z*.   $ x' = g(z) $
3. **Reconstruction Error**: The autoencoder is trained to minimize the reconstruction error, which is the difference between the original input *x* and the reconstructed output *x'*. A common choice for the loss function is the mean squared error (MSE):\   $ L(x, x') = ||x - x'||^2 $
4. **Anomaly Score**: The reconstruction error is used as the anomaly score. A high reconstruction error indicates that the data point is an anomaly.


## Python Implementation


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


### 1. Generate and Prepare Data


# Load data from the CSV file
data = pd.read_csv('../data/anomaly_data.csv')
X = data[['feature1', 'feature2']].values

# For autoencoders, we train on normal data only
# Let's assume the first 300 points are normal
X_train = X[:300]

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_scaled = scaler.transform(X)


### 2. Build the Autoencoder


input_dim = X.shape[1]
encoding_dim = 1

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')


### 3. Train the Autoencoder


autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)


### 4. Calculate Reconstruction Error


reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)


### 5. Identify Anomalies


# Set a threshold for anomaly detection
threshold = np.quantile(mse, 0.95)
anomalies = X[mse > threshold]


### 6. Visualize Results


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='b', label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', label='Anomalies')
plt.title('Autoencoder Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


## Pros & Cons


### Pros
- **Can learn complex patterns**: Autoencoders can capture complex, non-linear relationships in the data.
- **Effective for high-dimensional data**: They are well-suited for tasks like image and time-series anomaly detection.
- **No assumptions about data distribution**: They are a non-parametric method.

### Cons
- **Computationally expensive**: Training deep autoencoders can be time-consuming.
- **Requires a lot of data**: They typically require a large amount of normal data to learn effectively.
- **Can be difficult to tune**: The performance can be sensitive to the network architecture and hyperparameters.


## When to Use


Autoencoders are a good choice for anomaly detection when:
- The dataset is large and high-dimensional.
- The normal data has complex, non-linear patterns.
- There is a sufficient amount of normal data available for training.
