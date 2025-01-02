# imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# Import the data
data = pd.read_table(r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab2\BreastCancerData.txt') 


# Assuming 'BreastCancerData.txt' is in the current working directory or provide the full path
# Define column names based on the attribute information
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# Import the data with the correct headers
data = pd.read_csv('BreastCancerData.txt', sep=',', header=None, names=column_names)

#Print Data
print("Breast Cancer Data", data)

# Drop the 'Sample code number' attribute. Comment out line 24 when testing boxplot methods
#data = data.drop('Sample code number', axis=1)

# Display the first few rows of the DataFrame to verify
print(data.head())

# Replace '?' with NaN
data = data.replace('?', np.nan)


# Count the number of NaNs in each column
missing_values = data.isnull().sum()
print("Number of missing values in each column:")
print(missing_values)


# Drop rows with missing values
data = data.dropna()

# Verify the number of missing values again to confirm they are removed
print("Number of missing values after dropping rows:")
print(data.isnull().sum())  

# Check for duplicate instances
duplicates = data.duplicated()
num_duplicates = duplicates.sum() 
print(num_duplicates)

# Drop the row duplicates
data_cleaned = data.drop_duplicates()
print(data_cleaned)

# Randomly select 1% of the data without replacement
sampled_data = data.sample(frac=0.01, replace=False) 
print(sampled_data)


# Plot a histogram for 'Clump Thickness'
data['Clump Thickness'].hist(bins=10)
plt.title('Histogram of Clump Thickness')
plt.xlabel('Clump Thickness')
plt.ylabel('Frequency')
plt.show()

# Discretize 'Clump Thickness' into 4 bins of equal width
data['Clump Thickness Binned'] = pd.cut(data['Clump Thickness'], bins=4, labels=['Bin1', 'Bin2', 'Bin3', 'Bin4'])

# Output the range of values of each category and the number of records that belong to each of the categories
print(data['Clump Thickness Binned'].value_counts()) 
print(pd.cut(data['Clump Thickness'], bins=4).value_counts()) 

# Plotting the boxplot for all columns except 'Sample code number' to identify outliers
data.drop(['Sample code number'], axis=1).boxplot(figsize=(20, 10))
plt.xticks(rotation=45)  # Rotates the labels on the x-axis so they fit and are readable
plt.title('Boxplot of all attributes excluding sample code number')
plt.show()