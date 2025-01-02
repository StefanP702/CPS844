"""
Part 2: Anomaly Detection

Using the LocalOutlierFactor model for anomaly detection on the data from 'dataOutliers.npy'.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1) Load the data from the file 'dataOutliers.npy'
data = np.load('dataOutliers.npy')

# 2) Optionally, create a scatter plot to visualize the data
# plt.scatter(data[:,0], data[:,1], color='k', s=3)
# plt.show()

# 3) Anomaly detection: Density-based
# Fit the LocalOutlierFactor model for outlier detection
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
y_pred = lof.fit_predict(data)

# 4) Plot results
plt.figure(figsize=(10, 6))
# Scatter plot of the data
plt.scatter(data[:, 0], data[:, 1], color='k', s=3, label='Data Points')

# Indicate which points are outliers
outliers = data[y_pred == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], edgecolors='r', facecolors='none', s=30, label='Outliers')

plt.title('Anomaly Detection with LocalOutlierFactor')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
