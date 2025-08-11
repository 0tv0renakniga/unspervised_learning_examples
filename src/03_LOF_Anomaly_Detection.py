
# 3. Density-Based Anomaly Detection: Local Outlier Factor (LOF)


## Concept & Theory


Local Outlier Factor (LOF) is a density-based anomaly detection algorithm that measures the local deviation of the density of a given data point with respect to its neighbors. The core idea is to identify outliers by comparing the local density of a point to the local densities of its neighbors. Points in areas of lower density than their neighbors are considered outliers.


## Mathematical Overview


1. **k-distance**: The distance to the *k*-th nearest neighbor.2. **Reachability distance**: The reachability distance of a point *A* from a point *B* is the maximum of the k-distance of *B* and the true distance between *A* and *B*.   $$ \text{reach-dist}_k(A, B) = \max(\text{k-distance}(B), d(A, B)) $$
3. **Local reachability density (lrd)**: The inverse of the average reachability distance of a point *A* from its neighbors.   $$ \text{lrd}_k(A) = 1 / \left( \frac{\sum_{B \in N_k(A)} \text{reach-dist}_k(A, B)}{|N_k(A)|} \right) $$
4. **Local outlier factor (LOF)**: The ratio of the average lrd of the neighbors of *A* to the lrd of *A*.   $$ \text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}}{|N_k(A)|} = \frac{\sum_{B \in N_k(A)} \text{lrd}_k(B)}{|N_k(A)|} / \text{lrd}_k(A) $$An LOF score close to 1 means the point is in a dense region, while a score significantly greater than 1 indicates an outlier.


## Python Implementation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

### 1. Load Data

# Load data from the CSV file
data = pd.read_csv('../data/anomaly_data.csv')
X = data[['feature1', 'feature2']].values

### 2. Fit LOF Model

# novelty=True for outlier detection on new data
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=True).fit(X)

### 3. Identify Anomalies

# The predict method returns -1 for outliers and 1 for inliers.
y_pred = lof.predict(X)
anomalies = X[y_pred == -1]

### 4. Visualize Results

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Clusters')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', label='Anomalies')
plt.title('LOF Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

## Pros & Cons


### Pros
- **Effective in identifying local outliers.**: It can find outliers that have a different density than their neighbors.
- **No assumption about data distribution.**: It is a non-parametric method.
- **Works well with varying density clusters.**: It can handle clusters with different densities.

### Cons
- **Computationally expensive**: The calculation of the LOF score can be slow for large datasets.
- **Curse of dimensionality**: The distance metric can be less meaningful in high-dimensional spaces.
- **Parameter sensitive**: The choice of *k* (the number of neighbors) can be challenging.


## When to Use


LOF is a good choice for anomaly detection when:
- The dataset has clusters of varying densities.
- Outliers are expected to be local, meaning they are in regions of lower density than their neighbors.
- The dimensionality of the data is not excessively high.
