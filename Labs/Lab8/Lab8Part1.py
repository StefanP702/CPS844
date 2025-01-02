"""
Part 1
"""

# Imports
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load the dataset (DataLab8.csv)
moviesDataset = pd.read_csv('DataLab8.csv')

# 2. To perform a k-means analysis on the dataset, extract only the numerical attributes: remove the "user" attribute 
data = moviesDataset.drop('user', axis=1)

# 3. Create an empty list to store the SSE of each value of k
sse = []

# 4. Apply k-means with a varying number of clusters k and compute the corresponding sum of squared errors (SSE)
for k in range(1, 7):  # Considering there are 6 users, we try k from 1 to 6
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

# 5. Plot to find the SSE vs the Number of Clusters to visually find the "elbow"
plt.figure(figsize=(10, 6))
plt.plot(range(1, 7), sse, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# 6. Look at the plot and determine the number of clusters k
# You need to visually inspect the plot generated to determine the elbow point.
# For example, if the elbow seems to be at k=3, then:
k = 3

# 7. Using the optimized value for k, apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_

# 8. Display the assignments of each users to a cluster
clusters = pd.DataFrame(labels, index=moviesDataset['user'], columns=['Cluster ID'])
print(clusters)


