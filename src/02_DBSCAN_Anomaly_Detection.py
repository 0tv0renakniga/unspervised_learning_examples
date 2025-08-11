"""
# 2. Clustering-Based Anomaly Detection: DBSCAN
"""

"""
## Concept & Theory
"""

"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. The main idea is that a point is a core point if it has at least a minimum number of other points (MinPts) within a given radius (epsilon). Clusters are built by connecting core points that are close to each other. Points that are not part of any cluster are considered noise, which makes DBSCAN naturally suited for anomaly detection.
"""

"""
## Mathematical Overview
"""

"""
DBSCAN has two key parameters:
- **epsilon (eps)**: The radius of the neighborhood around a point.
- **MinPts**: The minimum number of points required to form a dense region.

Based on these parameters, points are classified as:
- **Core point**: A point that has at least MinPts points (including itself) in its eps-neighborhood.
- **Border point**: A point that is not a core point but is in the eps-neighborhood of a core point.
- **Noise point (Anomaly)**: A point that is neither a core point nor a border point. These are the points that are considered anomalies.
"""

"""
## Python Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

"""
### 1. Load Data
"""

# Load data from the CSV file
data = pd.read_csv('../data/anomaly_data.csv')
X = data[['feature1', 'feature2']].values

"""
### 2. Fit DBSCAN Model
"""

dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X)

"""
### 3. Identify Anomalies
"""

# Anomalies are labeled as -1 by DBSCAN
anomalies = X[dbscan.labels_ == -1]

"""
### 4. Visualize Results
"""

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis', label='Clusters')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', label='Anomalies')
plt.title('DBSCAN Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

"""
## Pros & Cons
"""

"""
### Pros
- **Can find arbitrarily shaped clusters.**: It is not limited to spherical clusters.
- **Does not require specifying the number of clusters.**: The algorithm finds the number of clusters on its own.
- **Robust to outliers**: DBSCAN is designed to identify and handle noise.

### Cons
- **Sensitive to parameters**: The choice of `eps` and `MinPts` can be difficult and has a large impact on the results.
- **Struggles with varying density**: It can be challenging to find a good set of parameters for clusters with varying densities.
- **Curse of dimensionality**: The distance metric can be less meaningful in high-dimensional spaces.
"""

"""
## When to Use
"""

"""
DBSCAN is a good choice for anomaly detection when:
- The clusters have complex shapes.
- The number of clusters is unknown.
- The dataset contains noise and outliers that need to be identified.
"""
