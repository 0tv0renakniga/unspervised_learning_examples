
# 4. Statistical Anomaly Detection: Gaussian Models


## Concept & Theory


The core idea behind using Gaussian models for anomaly detection is to fit a Gaussian distribution to the data. Data points that have a low probability under this distribution are considered anomalies. This method assumes that the normal data points are generated from a Gaussian distribution.


## Mathematical Overview


1. **Parameter Estimation**: We model the data using a multivariate Gaussian distribution with mean $\mu$ and covariance matrix $\Sigma$. These parameters are estimated from the data.\
   $$ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} $$$$\Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^T $$
2. **Probability Density Function**: The probability of a data point *x* is given by:\
   $$ p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right) $$
3. **Anomaly Detection**: We select a threshold $\epsilon$. If $p(x; \mu, \Sigma) < \epsilon$, the point is classified as an anomaly.


## Python Implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd


### 1. Load Data

# Load data from the CSV file
data = pd.read_csv('../data/anomaly_data.csv')
X = data[['feature1', 'feature2']].values


### 2. Estimate Parameters

mu = np.mean(X, axis=0)
sigma = np.cov(X.T)


### 3. Calculate Probabilities

p = multivariate_normal.pdf(X, mean=mu, cov=sigma)


### 4. Identify Anomalies

# Set a threshold for anomaly detection
threshold = 1e-3
anomalies = X[p < threshold]


### 5. Visualize Results

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='b', label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', label='Anomalies')
plt.title('Gaussian Model Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


## Pros & Cons


### Pros
- **Computationally efficient**: The model is fast to train and evaluate.
- **Probabilistic approach**: It provides a probability score for each data point.
- **Easy to interpret**: The model is based on a well-understood statistical distribution.

### Cons
- **Assumes Gaussian distribution**: It performs poorly if the data is not Gaussian.
- **Sensitive to outliers in training data**: The parameter estimation can be skewed by outliers.
- **May not work well with multimodal distributions**: A single Gaussian may not be a good fit for data with multiple clusters.


## When to Use


Gaussian models are a good choice for anomaly detection when:
- The data is known to follow a Gaussian distribution.
- The dataset is not too large, and the number of features is not too high.
- A probabilistic interpretation of the anomaly score is desired.
