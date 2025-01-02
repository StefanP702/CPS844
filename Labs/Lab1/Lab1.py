# imports
import pandas as pd

# Path to the iris.data.txt file
file_path = r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab1\iris.data.txt'

# Load the data into a DataFrame, assuming that the file is a CSV with a comma as a separator.
# If the file uses a different separator, you will need to specify it in the 'sep' parameter.
iris_df = pd.read_csv(file_path, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Calculating statistics only for numeric columns
numeric_cols = iris_df.columns[:-1]# This excludes the last column which is 'species'

print("Iris DataFrame:\n", iris_df)

print("\nStatistics for each quantitative attribute:")
print("Mean:\n", iris_df[numeric_cols].mean())
print("Standard Deviation:\n", iris_df[numeric_cols].std())
print("Minimum:\n", iris_df[numeric_cols].min())
print("Maximum:\n", iris_df[numeric_cols].max())

# Frequency count for each distinct class value
print("\nFrequency count for each distinct class value:")
print(iris_df['species'].value_counts())
