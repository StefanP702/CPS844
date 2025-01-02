# import any package you need
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# Import the data

data = pd.read_table(r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab3\vertebrate.csv')  
data = pd.read_csv('vertebrate.csv', header=0)

# 1. Convert the data into a binary classification: mammals vs non-mammals
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'], 'non-mammals')

# 2. Keep only the attributes of interest and separate the target class
attributes = ['Warm-blooded', 'Gives Birth', 'Aquatic Creature', 'Aerial Creature', 'Has Legs', 'Hibernates']
X = data[attributes]
y = data['Class']

# 4. Create a decision tree classifier object
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# 5. Train the classifier
classifier.fit(X, y)

# 6. Prepare the test data and apply the decision tree to classify the test records
testData = pd.DataFrame([
    ['lizard', 0, 0, 0, 0, 1, 1, 'non-mammals'],
    ['monotreme', 1, 0, 0, 0, 1, 1, 'mammals'],
    ['dove', 1, 0, 0, 1, 1, 0, 'non-mammals'],
    ['whale', 1, 1, 1, 0, 0, 0, 'mammals']
], columns=data.columns)

X_test = testData[attributes]
y_test = testData['Class']

# Extract the class attributes and target class from 'testData'
predictions = classifier.predict(X_test)

# 7. Compute and print out the accuracy of the classifier on 'testData'
accuracy = sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy}')

# 8. Plot your decision tree
plt.figure(figsize=(12,8))
plot_tree(classifier, filled=True, feature_names=attributes, class_names=['non-mammals', 'mammals'])
plt.show()
