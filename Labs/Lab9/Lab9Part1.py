"""
Part 1: Density-Based Clustering methods

Using DBSCAN to identify high-density clusters separated by regions of low density.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Import the chameleon.data data
data = pd.read_csv('chameleon.data', delimiter=' ', names=['x','y'])

# Check the data distribution
plt.figure(figsize=(10, 6))
data.plot.scatter(x='x', y='y')
plt.title('Data Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Apply DBScan with eps=15.5 and minpts=5
dbscan_analysis = DBSCAN(eps=15.5, min_samples=5).fit(data)

# Convert labels to a pandas dataframe
clusters_labels = pd.DataFrame(dbscan_analysis.labels_, columns=['Cluster ID'])

# Concatenate the dataframes 'data' and 'clusters_labels'
result = pd.concat([data, clusters_labels], axis=1)

# Create a scatter plot of the data
plt.figure(figsize=(10, 6))
plt.scatter(result['x'], result['y'], c=result['Cluster ID'], cmap='jet', marker='.')
plt.title('DBSCAN Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Cluster ID')
plt.show()

# Count the number of clusters (excluding noise)
num_clusters = len(result['Cluster ID'].unique()) - (1 if -1 in result['Cluster ID'].values else 0)
print(f'There are {num_clusters} clusters, not including the noise')
