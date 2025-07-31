# Hierarchical Clustering

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("cleaned_data.csv")
scaler = StandardScaler()

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_sample = df.sample(n=1000, random_state=17)

plt.figure(figsize=(10, 7))
dendrogram(linkage(df_sample[numeric_columns], method='ward')) # ward because it minimizes the variance within each cluster
plt.title('Hierarchical Clustering (sampled)')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()