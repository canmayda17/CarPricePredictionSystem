import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns



# load the data

df = pd.read_csv("data.csv")


# clean the numerical data
df["fuel_consumption_l_100km"] = (
    df["fuel_consumption_l_100km"]
    .astype(str)
    .str.replace(",", ".")
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)

df["mileage_in_km"] = pd.to_numeric(df["mileage_in_km"], errors='coerce')
df["price_in_euro"] = pd.to_numeric(df["price_in_euro"], errors='coerce')
df["power_kw"] = pd.to_numeric(df["power_kw"], errors='coerce')
df["year"] = pd.to_numeric(df["year"], errors='coerce')


# create a 10% sample
df_sample = df.sample(frac=0.1, random_state=42).copy()


# feature selection and scaling
numeric_columns = ["year", "price_in_euro", "power_kw", "fuel_consumption_l_100km", "mileage_in_km"]
features = df_sample[numeric_columns].copy()

for col in numeric_columns:
    mean_val = features[col].mean()
    features[col].fillna(mean_val, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


# apply DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
df_sample["cluster"] = dbscan.fit_predict(X_scaled)



print("Cluster counts:")
print(df_sample["cluster"].value_counts())


# cluster statistics (mean & median)
print("\nCluster-wise mean and median values:")
valid_clusters = df_sample[df_sample["cluster"] != -1]
cluster_summary = valid_clusters.groupby("cluster")[numeric_columns].agg(["mean", "median"]).round(2)
print(cluster_summary)


# visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_sample,
    x="price_in_euro",
    y="mileage_in_km",
    hue="cluster",
    palette="tab10",
    legend="full"
)
plt.title("DBSCAN Clustering (10% sample)")
plt.xlabel("Price (€)")
plt.ylabel("Mileage (km)")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
