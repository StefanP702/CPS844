import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the dataset
file_path = (r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab6\weather.csv')  
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head(),"\n")

# Convert categorical variables into dummy variables and ensure they're of type 'float'
data_dummies = pd.get_dummies(data, columns=['outlook', 'windy', 'play'], drop_first=False, dtype=float)

# Drop the 'play_no' target attribute
data_dummies = data_dummies.drop(['play_no'], axis=1)

# Separate the features from the target attribute
X = data_dummies.drop(['play_yes'], axis=1)
y = data_dummies['play_yes']

# Construct and train a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X, y)

# Attributes for the new day
new_day = pd.DataFrame({
    'temperature': [66],
    'humidity': [90],
    'outlook_overcast': [0],
    'outlook_rainy': [0],
    'outlook_sunny': [1],
    'windy_False': [0],
    'windy_True': [1]
})

# Predict the likelihood of play = yes for the new day
new_day_pred_proba = gnb.predict_proba(new_day)

# Extract and print the likelihoods
likelihood_yes, likelihood_no = new_day_pred_proba[0][1], new_day_pred_proba[0][0]
print(f"Likelihood of play = yes: {likelihood_yes}")
print(f"Likelihood of play = no: {likelihood_no}")

