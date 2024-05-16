import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


df=pd.read_csv('featureSelection/data.csv')
data = df.drop(columns=['Target'])*100

output_categories = df['Target']

num_bins = 4
data_categorical = pd.cut(data.stack(), bins=num_bins, labels=False).unstack()

# Initialize variables
num_output_categories = num_bins
num_features = data_categorical.shape[1]
g_test_results = []

# Iterate over each feature
for i in data.columns:
    # Create contingency table
    contingency_table = pd.crosstab(output_categories, data_categorical[i])
    
    # Perform G-test
    g_stat, p_val, _, _ = chi2_contingency(contingency_table, lambda_="log-likelihood")
    
    # Store results
    g_test_results.append((i, g_stat, p_val))

# Sort results based on p-value
sorted_results = sorted(g_test_results, key=lambda x: x[2])

# Print results
for i, result in enumerate(sorted_results):
    feature_index, g_stat, p_val = result
    print(f"Feature {feature_index}: G-statistic={g_stat}, p-value={p_val}")

# Example of interpreting results:
# The smaller the p-value, the more significant the association between the feature and output categories.
# Typically, if the p-value is below a certain significance level (e.g., 0.05), we reject the null hypothesis 
# and conclude that there is a significant association between the feature and output categories.
