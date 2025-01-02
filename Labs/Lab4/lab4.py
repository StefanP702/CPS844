import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

pathX = (r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab4\Xdata.npy')
pathY = (r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab4\Ydata.npy')

# 1) (10 points) Load the data (Y is the class labels of X)
X = np.load(pathX)
Y = np.load(pathY)


# Split the training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Initialize lists to store accuracies and depths
train_accuracies = []
test_accuracies = []
depths = range(1, 51)

# Loop over the depths
for depth in depths:
    # Create and train the decision tree
    tree_classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_classifier.fit(X_train, Y_train)
    
    # Make predictions
    Y_train_pred = tree_classifier.predict(X_train)
    Y_test_pred = tree_classifier.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    
    # Store accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot of training and test accuracies vs the tree depths
plt.figure(figsize=(10, 5))
plt.plot(depths, train_accuracies, 'rv-', depths, test_accuracies, 'bo--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.grid(True)

# Show the plot
plt.show()

# Determine overfitting depth
overfitting_depth = None
max_test_accuracy = max(test_accuracies)
for depth, accuracy in zip(depths, test_accuracies):
    if accuracy < max_test_accuracy:
        overfitting_depth = depth
        break

overfitting_depth
