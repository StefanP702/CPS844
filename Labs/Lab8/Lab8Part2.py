"""
Part 2
"""

# Imports
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 1. Import the vertebrate.csv data
data = pd.read_csv('vertebrate.csv')

# 2. Pre-process data: create a new variable and bind it with all the numerical attributes (i.e., all except the 'Name' and 'Class')
NumericalAttributes = data.drop(['Name', 'Class'], axis=1)

# 3. Single link (MIN) analysis + plot associated dendrogram 
min_analysis = linkage(NumericalAttributes, method='single')

# 4. Plot the associated dendrogram. 
plt.figure(figsize=(10, 8))
dendrogram(min_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Dendrogram for Single Link (MIN) Analysis')
plt.show()

# 5. Complete Link (MAX) analysis + plot associated dendrogram 
max_analysis = linkage(NumericalAttributes, method='complete')

# 6. Plot the associated dendrogram. 
plt.figure(figsize=(10, 8))
dendrogram(max_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Dendrogram for Complete Link (MAX) Analysis')
plt.show()

# 7. Group Average analysis 
average_analysis = linkage(NumericalAttributes, method='average')

# 8. Plot the associated dendrogram. 
plt.figure(figsize=(10, 8))
dendrogram(average_analysis, labels=data['Name'].tolist(), orientation='right')
plt.title('Dendrogram for Group Average Analysis')
plt.show()

