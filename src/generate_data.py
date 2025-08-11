import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Generate isotropic Gaussian blobs for clustering
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Add some outliers
outliers = np.array([[0, 5], [6, 2]])
X = np.concatenate([X, outliers])

# Create a pandas DataFrame
df = pd.DataFrame(X, columns=['feature1', 'feature2'])

# Save to CSV
df.to_csv('/home/scotty/unspervised_learning_examples/data/anomaly_data.csv', index=False)