import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted')
    }

# Function to print metrics
def print_metrics(metrics):
    for metric, value in metrics.items():
        print(f"- {metric.capitalize()}: {value}")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title(title)
    plt.show()

# Function to evaluate and print model performance
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics_train = calculate_metrics(y_train, train_pred)
    metrics_test = calculate_metrics(y_test, test_pred)
    print(f"Model performance for {name} - Training set")
    print_metrics(metrics_train)
    print("-----")
    print(f"Model performance for {name} - Test set")
    print_metrics(metrics_test)
    plot_confusion_matrix(y, model.predict(X), f"{name} Confusion Matrix")

# Load the dataset
b1 = pd.read_csv('Iris.data')
b1 = b1.dropna()
print("Data set\n",b1)

# Display basic information
print(b1.isnull().sum())
print(b1.info())

# Visualize data distributions
for col in b1.columns[:-1]:
    plt.hist(b1[col])
    plt.title(col)
    plt.show()

# Display value counts for the 'class' column
print(b1.value_counts('class'))

# Prepare the data
X = b1.drop('class', axis=1)
y = b1['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store performance metrics
accuracy_train, mcc_train, f1_train, precision_train, recall_train = [], [], [], [], []
accuracy_test, mcc_test, f1_test, precision_test, recall_test = [], [], [], [], []

# KNN Classifier or KNeighborsClassifier
knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)
Knn_train_pred, Knn_test_pred = knn.predict(X_train), knn.predict(X_test)
evaluate_model("KNN", knn, X_train, y_train, X_test, y_test)

# SVM Classifier
svm_rbf = SVC()
svm_rbf.fit(X_train, y_train)
Svm_train_pred, Svm_test_pred = svm_rbf.predict(X_train), svm_rbf.predict(X_test)
evaluate_model("SVM", svm_rbf, X_train, y_train, X_test, y_test)

# Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
Dt_train_pred, Dt_test_pred = dt.predict(X_train), dt.predict(X_test)
evaluate_model("Decision Tree", dt, X_train, y_train, X_test, y_test)

# Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train, y_train)
NB_train_pred, NB_test_pred = NB.predict(X_train), NB.predict(X_test)
evaluate_model("Naive Bayes", NB, X_train, y_train, X_test, y_test)

# MLP Classifier
mlp = MLPClassifier(alpha=1, max_iter=1000)
mlp.fit(X_train, y_train)
Mlp_train_pred, Mlp_test_pred = mlp.predict(X_train), mlp.predict(X_test)
evaluate_model("Neural Network/MLP", mlp) 

