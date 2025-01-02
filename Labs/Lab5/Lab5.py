import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv(r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab5\breast-cancer-wisconsin.data', header=None)

# Assign new headers to the DataFrame
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data.columns = column_names

# Pre-process the data
data = data.drop(['Sample code number'], axis=1)
data = data.replace('?', np.nan)
data = data.dropna()
data['Bare Nuclei'] = pd.to_numeric(data['Bare Nuclei'])
data = data.drop_duplicates()

# Separate the features and the target class
X = data.drop(['Class'], axis=1)
y = data['Class'].replace({2: 0, 4: 1})  # Map target values

# Standardize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Split the data into training and test sets for confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Construct Nearest Neighbors classifier
knn = KNeighborsClassifier()

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Scoring methods for cross-validation
scoring_methods = ['accuracy', 'f1', 'precision', 'recall']

# Compute metrics using cross-validation with correct scoring methods
results_corrected = {method: cross_val_score(knn, X_standardized, y, cv=kf, scoring=method).mean() for method in scoring_methods}

# Train the classifier on the training set and predict on the test set
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print out the averages of the accuracies, f1-scores, precisions, and recall measurements
print("Average metrics from 10-fold cross-validation:")
for metric, score in results_corrected.items():
    print(f"{metric.capitalize()}: {score:.4f}")

# Displaying the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
