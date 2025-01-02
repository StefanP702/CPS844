import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from apyori import apriori

# Function for Clustering Analysis
def perform_clustering(dataframe):
    # Selecting numerical features for clustering
    features_for_clustering = dataframe.drop(['Class'], axis=1)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(features_for_clustering)
    
    # Add cluster labels to the dataframe
    dataframe['Cluster'] = kmeans.labels_
    
    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['MajorAxisLength'], dataframe['MinorAxisLength'], c=dataframe['Cluster'], cmap='viridis')
    plt.title('Clustering of Raisin Dataset based on Major and Minor Axis Length')
    plt.xlabel('MajorAxisLength')
    plt.ylabel('MinorAxisLength')
    plt.colorbar(label='Cluster')
    plt.show()
    
    return dataframe

# Function for Association Analysis (adapted for numerical data)
def perform_association_analysis(dataframe):
    # Discretize numerical features into categories
    dataframe['Area'] = pd.cut(dataframe['Area'], bins=3, labels=['Small', 'Medium', 'Large'])
    dataframe['Perimeter'] = pd.cut(dataframe['Perimeter'], bins=3, labels=['Short', 'Medium', 'Long'])
    # Convert to list of lists for apriori
    transactions = dataframe.drop(['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'Extent', 'Class', 'Cluster'], axis=1)
    transactions = transactions.values.tolist()
    
    # Perform association analysis with less strict thresholds
    association_rules = apriori(transactions, min_support=0.05, min_confidence=0.1, min_lift=1.5)
    rules_list = list(association_rules)
    
    # Check if rules were found
    if not rules_list:
        print("No association rules were found.")
    else:
        # Print out the rules
        print("Association Rules:")
        for rule in rules_list:
            # Get the items in the rule
            items = [item for item in rule.items]
            print(f"Rule: {items}")
            print(f"Support: {rule.support}")
            for ordered_stat in rule.ordered_statistics:
                print(f"Confidence: {ordered_stat.confidence}")
                print(f"Lift: {ordered_stat.lift}")
            print("\n")  # New line for readability
            
# Load the dataset
raisin_data_path = 'Raisin_Dataset.xlsx'
raisin_data = pd.read_excel(raisin_data_path)


# Display the first few rows of the dataframe to understand its structure
printa = raisin_data.head() 

print(printa)

# Perform Clustering Analysis
raisin_data = perform_clustering(raisin_data)

# Perform Association Analysis
perform_association_analysis(raisin_data)
