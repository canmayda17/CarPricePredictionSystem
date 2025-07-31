import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# Read data from CSV

df = pd.read_csv("data.csv")  # Change the file name according to your data


#  Convert non-numeric columns (e.g., prices to numeric)
# Clean the 'price_in_euro' column and convert it to float
df['price_in_euro'] = df['price_in_euro'].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0]
df['price_in_euro'] = pd.to_numeric(df['price_in_euro'], errors='coerce')

# Clean the 'mileage_in_km' and 'year' columns
df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')



# Convert categorical variables into ranges
df['price_range'] = pd.cut(df['price_in_euro'], bins=[0, 10000, 20000, 30000, 50000, 1000000],
                           labels=['low', 'mid-low', 'mid', 'mid-high', 'high'])

df['year_range'] = pd.cut(df['year'], bins=[1900, 2005, 2015, 2019, 2022, 2025],
                          labels=['old', 'mid-old', 'recent', 'new', 'very new'])

df['mileage_range'] = pd.cut(df['mileage_in_km'], bins=[0, 25000, 50000, 100000, 200000, 1000000],
                             labels=['very low', 'low', 'medium', 'high', 'very high'])



# Convert categorical data to one-hot encoding
df_encoded = pd.get_dummies(df[['brand', 'model', 'color', 'transmission_type', 'fuel_type', 'price_range', 'year_range', 'mileage_range']])


# Find frequent itemsets using Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)


# Generate association rules from Apriori
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)


#  Filter interesting rules (e.g., lift > 1.2 and confidence > 0.6)
interesting_rules = rules[(rules["lift"] > 1.2) & (rules["confidence"] > 0.6)]


print("\n Interesting Apriori Rules:")
print(interesting_rules[["antecedents", "consequents", "support", "confidence", "lift"]])
