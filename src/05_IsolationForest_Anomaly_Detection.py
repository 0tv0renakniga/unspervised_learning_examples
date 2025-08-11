
# 5. Model-Based Anomaly Detection: Isolation Forest


## Concept & Theory


Isolation Forest is an ensemble-based anomaly detection algorithm. The core idea is that anomalies are "few and different" and therefore easier to isolate than normal points. The algorithm builds an ensemble of isolation trees (iTrees) for the data. To isolate a data point, the algorithm recursively partitions the data until the point is isolated. The number of partitions required to isolate a point is its path length. Anomalies are expected to have shorter path lengths than normal points.


## Mathematical Overview


1. **iTree Construction**: An iTree is a binary tree where each node represents a partition of the data. A random feature and a random split value are chosen to partition the data at each node.\
2. **Path Length**: The path length *h(x)* of a point *x* is the number of edges from the root of the iTree to the node where *x* is isolated.\
3. **Anomaly Score**: The anomaly score *s(x, n)* for a point *x* is calculated as:\
   $$ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} $$
   where *E(h(x))* is the average path length of *x* over all iTrees, and *c(n)* is the average path length of an unsuccessful search in a Binary Search Tree, given by:\
   $$ c(n) = 2H(n-1) - (2(n-1)/n) $$
   where *H(i)* is the harmonic number, which can be estimated as *ln(i) + 0.5772156649* (Euler's constant).\
   - If the score is close to 1, the point is likely an anomaly.\
   - If the score is less than 0.5, the point is likely a normal point.\
   - If all scores are close to 0.5, then the entire sample does not seem to have any distinct anomalies.


## Python Implementation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd


### 1. Load Data

# Load data from the CSV file
data = pd.read_csv('../data/anomaly_data.csv')
X = data[['feature1', 'feature2']].values


### 2. Fit Isolation Forest Model

# contamination='auto' is a good default
iso_forest = IsolationForest(contamination='auto', random_state=0).fit(X)


### 3. Identify Anomalies

# The predict method returns -1 for outliers and 1 for inliers.
y_pred = iso_forest.predict(X)
anomalies = X[y_pred == -1]


### 4. Visualize Results

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Clusters')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='r', label='Anomalies')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


## Pros & Cons


### Pros
- **Efficient**: It has a low time complexity and is suitable for large datasets.
- **Handles high-dimensional data well**: It can be effective even with a large number of features.
- **No need to specify the number of clusters**: It does not require any cluster-related parameters.

### Cons
- **Can be sensitive to the number of trees**: The performance can be affected by the number of iTrees in the forest.
- **May not perform well on complex datasets**: It may struggle with datasets that have complex structures and no clear separation between normal and anomalous points.


## When to Use


Isolation Forest is a good choice for anomaly detection when:
- The dataset is large and high-dimensional.
- There is no prior knowledge about the data distribution.
- Anomalies are expected to be few and different from normal data.
