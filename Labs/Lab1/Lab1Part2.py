# pip install pandas

# imports
import pandas as pd

# Define the file path
file_path = (r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab1\iris.data.txt')

# Define the column names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the data into a DataFrame
iris_df = pd.read_csv(file_path, header=None, names=columns)

# Print the entire DataFrame
print(iris_df)
