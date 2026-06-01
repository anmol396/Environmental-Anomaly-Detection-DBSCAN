# DBSCAN-Based Detection of Environmental Anomalies in Ahmedabad (2023–2025)

import pandas as pd
df = pd.read_csv("pollution_data.csv")
df

df.shape

# Data Information

df.info()

df.describe()

# #Checking for Nans in the whole data

print(df.isna().sum())

# Data Preprocessing

col = ['PM2.5 (µg/m³)','PM10 (µg/m³)','NO₂ (µg/m³)','SO₂ (µg/m³)','CO (mg/m³)','O₃ (µg/m³)']

df[col].isna().all(axis=1).sum()

# Remove rows where all pollutant values are missing

df1 = df.dropna(subset=col,how="all")


print(df1.isna().sum())

df1[col].dtypes

df1 = df1.copy()

# Converting Date column to datetime format 

df1["Date"] = pd.to_datetime(df1["Date"])

df1 = df1.sort_values(["Station_ID", "Date"])

df1 = df1.set_index("Date")

# Time interpolation for short gaps
df1[col] = (df1.groupby('Station_ID')[col].transform(lambda x: x.interpolate(method='time', limit=3)))

# Applying forward/backward fill for long gaps 
df1[col] = df1.groupby('Station_ID')[col].ffill().bfill()
df1[col].isna().sum()


df1=df1.drop(columns=['Station_ID','City']) #both of these are not contributing anything to the output

# Feature Scaling
# DBSCAN is distance-based, so pollutant features should be scaled which ensures all pollutants contribute equally to clustering.
# Without scaling, the pollutants' contribution is very different and misleading
import seaborn as sns

sns.kdeplot(df1[col]);

# RobustScaler is used to normalize air pollution features
from sklearn.preprocessing import RobustScaler
x=df1[col]
sc=RobustScaler()
x_scaled=sc.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled,columns=col,index=df1.index)
# Now the data is fully scaled
import seaborn as sns
sns.kdeplot(x_scaled[col]);


# DBSCAN Parameter Selection
# Two key parameters:
# - eps: neighborhood radius.It is chosen using k-distance elbow method
# - min_samples: minimum points to form a dense cluster

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

neighbors = NearestNeighbors()
n_fit = neighbors.fit(x_scaled)
distances, indices = n_fit.kneighbors(x_scaled)

distances = distances[:, 4]
distances = sorted(distances)

plt.plot(distances)
plt.ylabel("5-NN distance")
plt.xlabel("Points sorted by distance")
plt.title("k-distance Graph for DBSCAN")
plt.show()


# DBSCAN Clustering

from sklearn.cluster import DBSCAN

for eps in [1.4, 1.5, 1.6, 1.7, 1.8]:
    db = DBSCAN(eps=eps, min_samples=7)
    labels = db.fit_predict(x_scaled)
    print(f"eps={eps}, anomalies={(labels == -1).sum()}")
#as alll the values of eps near the elbow point are showing similar kind of results, we can say that DBSCAN works best at eps=1.6
#min_samples is good when it is >= no. of features+1


dbscan = DBSCAN(eps=1.5, min_samples=7)
labels = dbscan.fit_predict(x_scaled)
df1['Cluster'] = labels
df1['Cluster'].value_counts()

# DBSCAN Clustering and Anomaly Detection
# DBSCAN clusters dense pollution patterns as normal behavior.
# Points labeled as `-1` are treated as environmental anomalies.


anomalies=df1[df1['Cluster']==-1]
normal=df1[df1['Cluster']!=-1]


df1["Anomaly/Normal"] = (df1['Anomaly/Normal'] = df1['Cluster'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
print(df1)

# Results and Visualizations

import plotly.express as px

fig = px.scatter_3d(
df1,x="PM2.5 (µg/m³)",y="PM10 (µg/m³)",z="NO₂ (µg/m³)",color="Anomaly/Normal",title="3D DBSCAN Clustering")
fig.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(
    df1[col + ['Anomaly/Normal']],
    hue='Anomaly/Normal',
    palette={'Normal': 'blue', 'Anomaly': 'red'},
    diag_kind='kde',
    plot_kws={'alpha': 0.6, 's': 30}
)
plt.suptitle('DBSCAN Pair Plot – CPCB Air Quality Data', y=1.02)
plt.show()

import plotly.express as px

fig = px.scatter(
    df1,
    x=df1.index,
    y="PM2.5 (µg/m³)",
    color=df1['Anomaly/Normal'].astype(str),
    title="PM2.5 Over Time with DBSCAN Anomalies Highlighted",
    opacity=0.6
)

fig.show()

import pandas as pd

# Station Coordinates (Ahmedabad Monitoring Stations)

station_coords = pd.DataFrame({
"Station_Name": ["Chandkheda","Gyaspur","Maninagar","Raikhad","Rakhial","SAC ISRO Bopal","SAC ISRO Satellite",
                 "SVPI Airport Hansol","Sardar Vallabhai Patel Stadium"],


"Latitude": [ 23.11, 22.97, 23.00, 23.03, 23.02, 23.03, 23.02, 23.07, 23.04],

"Longitude": [  72.58,  72.63,  72.60,  72.58,  72.61,72.47,  72.50,  72.63,  72.57] 
})

station_coords

# Percentage of Detected Anomalies

num_anomalies = (df1["Cluster"].value_counts().get(-1, 0))

total_points = len(df1)

percent_anomalies = (num_anomalies/ total_points) * 100

print(f"Percentage of anomalies: {percent_anomalies:.2f}%")

# Station-wise Anomaly Count

anomalies = (df1[df1["Cluster"] == -1])

anomalies["Station_Name"].value_counts()

# Monthly Distribution of Anomalies

df1.index[
df1["Cluster"] == -1
].month.value_counts()

# Model Validation

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Silhouette Score

# Higher = Better Clustering

mask = (df1["Cluster"] != -1)

score = silhouette_score(x_scaled[mask],df1.loc[mask,"Cluster"])

print("Silhouette Score:",score)

# Davies–Bouldin Index

# Lower = Better Clustering

dbi = davies_bouldin_score(
x_scaled[mask],df1.loc[mask,"Cluster"])

print("Davies-Bouldin Index:",dbi)

# Calinski–Harabasz Score

# Higher = Better Separation

chs = (calinski_harabasz_score(x_scaled[mask],df1.loc[mask,"Cluster"]))

print("Calinski-Harabasz Score:",chs)

# Noise Ratio

noise_ratio = (df1["Cluster"] == -1).mean()

print("Noise Ratio:",noise_ratio)

# Stability Analysis

print("Stability Check:")

for eps in [1.5,1.6,1.7]:
    labels = DBSCAN( eps=eps, min_samples=5).fit_predict( x_scaled)
    print( eps, (labels == -1).sum())


# Station-wise Cluster Summary

df1.groupby(["Station_Name","Cluster"]).size().reset_index(name="Count")

# Statistical Summary of Anomalies

anomalies[col].describe()

# Graphs

# Monthly Environmental Anomalies

anomalies = (df1[df1["Cluster"] == -1]).reset_index()

anomaly_month_count = (anomalies.groupby(anomalies["Date"].dt.to_period("M")).size().reset_index(name="Anomaly_Count"))

anomaly_month_count["Date"] = (anomaly_month_count["Date"].astype(str))

fig = px.line(anomaly_month_count,x="Date",y="Anomaly_Count",markers=True,title="Monthly Environmental Anomalies Over Time")

fig.show()

# PCA Visualization

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = (pca.fit_transform(x_scaled))

df1["PC1"] = (X_pca[:, 0])

df1["PC2"] = (X_pca[:, 1])

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

print("Total Variance Captured:", pca.explained_variance_ratio_.sum())

# Station Mapping

df_map = (df1.merge(station_coords,on="Station_Name",how="left"))

df_map["Point_Type"] = (df_map["Cluster"].apply(lambda x:"Anomaly"if x == -1else "Normal"))

fig = px.scatter_map(
df_map,
lat="Latitude",
lon="Longitude",
color="Point_Type",
hover_name="Station_Name",
zoom=10,
opacity=0.6
title="Normal vs Anomaly Pollution Points Across Ahmedabad")

fig.update_layout(mapbox_style="open-street-map")
fig.show()
