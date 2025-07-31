# Z-score

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cleaned_data.csv")

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("Original Values for Mean ve Std:\n")
print(df[numeric_columns].agg(['mean', 'std']))

print("\nZ-score Normalization for First 5 Rows:\n")
print(df_scaled[numeric_columns].head())

