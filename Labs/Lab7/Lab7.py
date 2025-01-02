import pandas as pd
from apyori import apriori

# Load the dataset
weather_data = pd.read_csv(r'C:\Users\StefW10\Desktop\Courses\CPS 844\Labs\Lab7\weather.csv')  

# Discretize the 'temperature' into 3 equal-width bins: 'cool', 'mild', 'hot'
weather_data['temperature'] = pd.cut(weather_data['temperature'], bins=3, labels=['cool', 'mild', 'hot'])

# Discretize the 'humidity' into 2 equal-width bins: 'normal', 'high'
weather_data['humidity'] = pd.cut(weather_data['humidity'], bins=2, labels=['normal', 'high'])

# Convert boolean values from 'windy' to string
weather_data['windy'] = weather_data['windy'].map({False: 'False', True: 'True'})

# Convert the whole dataset to a list of lists
transactions = [list(row) for row in weather_data.values]

# Perform association analysis using the apriori algorithm
# Start with a minimum support threshold of 0.28 and a minimum confidence threshold of 0.5
association_rules = apriori(transactions, min_support=0.28, min_confidence=0.5)

# Print out each of the rules generated along with their corresponding support and confidence values
for itemset in association_rules:
    for rule_index in range(len(itemset.ordered_statistics)):
        base = list(itemset.ordered_statistics[rule_index].items_base)
        add = list(itemset.ordered_statistics[rule_index].items_add)
        support = itemset.support
        confidence = itemset.ordered_statistics[rule_index].confidence
        print(f"Rule: {base} -> {add}\nSupport: {support}\nConfidence: {confidence}\n")
