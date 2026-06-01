#!/usr/bin/env python
# coding: utf-8

# # DBSCAN-Based Detection of Environmental Anomalies in Ahmedabad (2023–2025)
# 
# This project applies the DBSCAN clustering algorithm to identify unusual air pollution events (anomalies) using CPCB monitoring data from multiple stations in Ahmedabad.
# 

# ## Objective
# 
# The main goal of this project is to detect abnormal pollution spikes using an unsupervised machine learning approach.
# 
# We aim to:
# - Cluster normal air quality patterns
# - Identify rare pollution events as anomalies
# - Analyze seasonal and station-wise anomaly trends

# ## Dataset Description
# 
# Source: CPCB CAAQMS Air Quality Monitoring Data  
# Location: Ahmedabad, Gujarat  
# Duration: 2023–2025  
# Stations: 9 monitoring sites
# 
# Pollutants included:
# - PM2.5, PM10, NO₂, SO₂, CO, O₃

# # Loading the dataset

# In[34]:


import pandas as pd
df=pd.read_csv("pollution_data.csv")
df


# In[36]:


df.shape


# In[38]:


# Data informtion
df.info()


# In[40]:


df.describe()


# In[42]:


#Checking for Nans in the whole data
print(df.isna().sum())


# ## Data Preprocessing

# In[45]:


col=['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO₂ (µg/m³)', 'SO₂ (µg/m³)',	'CO (mg/m³)', 'O₃ (µg/m³)']
df[col].isna().all(axis=1).sum()


# In[47]:


#Removing rows where all pollutant values are missing
df1=df.dropna(subset=col,how="all")


# In[49]:


print(df1.isna().sum())


# In[51]:


df1[col].dtypes


# In[53]:


df1 = df1.copy()
# Converting Date column to datetime format 
df1['Date'] = pd.to_datetime(df1['Date'])

df1 = df1.sort_values(['Station_ID', 'Date'])
df1 = df1.set_index('Date')

# Time interpolation for short gaps
df1[col] = (df1.groupby('Station_ID')[col].transform(lambda x: x.interpolate(method='time', limit=3)))

# Applying forward/backward fill for long gaps 
df1[col] = df1.groupby('Station_ID')[col].ffill().bfill()


# In[55]:


df1[col].isna().sum()


# In[57]:


df1=df1.drop(columns=['Station_ID','City']) #both of these are not contributing anything to the output


# In[59]:


df1


# # Feature Scaling
# 
# DBSCAN is distance-based, so pollutant features should be scaled which ensures all pollutants contribute equally to clustering.

# In[63]:


# Without scaling, the pollutants' contribution is very different and misleading
import seaborn as sns
sns.kdeplot(df1[col]);


# In[65]:


# RobustScaler is used to normalize air pollution features
from sklearn.preprocessing import RobustScaler
x=df1[col]
sc=RobustScaler()
x_scaled=sc.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled,columns=col,index=df1.index)


# In[67]:


# Now the data is fully scaled
import seaborn as sns
sns.kdeplot(x_scaled[col]);


# # DBSCAN Parameter Selection
# 
# Two key parameters:
# - eps: neighborhood radius
# - min_samples: minimum points to form a dense cluster

# In[70]:


# eps is chosen using the k-distance elbow method.

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


# In[72]:


from sklearn.cluster import DBSCAN


# In[74]:


for eps in [1.4, 1.5, 1.6, 1.7, 1.8]:
    db = DBSCAN(eps=eps, min_samples=7)
    labels = db.fit_predict(x_scaled)
    print(f"eps={eps}, anomalies={(labels == -1).sum()}")


# In[76]:


#as alll the values of eps near the elbow point are showing similar kind of results, we can say that DBSCAN works best at eps=1.6
#min_samples is good when it is >= no. of features+1


# In[78]:


dbscan = DBSCAN(eps=1.5, min_samples=7)
labels = dbscan.fit_predict(x_scaled)
df1['Cluster'] = labels
df1['Cluster'].value_counts()


# ## DBSCAN Clustering and Anomaly Detection
# 
# DBSCAN clusters dense pollution patterns as normal behavior.
# Points labeled as `-1` are treated as environmental anomalies.

# In[81]:


anomalies=df1[df1['Cluster']==-1]
normal=df1[df1['Cluster']!=-1]


# In[83]:


df1['Anomaly/Normal'] = df1['Cluster'].apply(
    lambda x: 'Anomaly' if x == -1 else 'Normal'
)


# In[85]:


df1


# # Results & Visualizations

# In[88]:


import plotly.express as px

fig = px.scatter_3d(
    df1,
    x='PM2.5 (µg/m³)',
    y='PM10 (µg/m³)',
    z='NO₂ (µg/m³)',
    color=df1['Anomaly/Normal'],
    title="3D DBSCAN Clustering (Anomalies Highlighted)",
    opacity=0.7
)

fig.show()


# In[89]:


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


# In[90]:


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


# In[ ]:





# In[93]:


import pandas as pd

# Station coordinate table (Ahmedabad)
station_coords = pd.DataFrame({
    "Station_Name": [
        "Chandkheda", "Gyaspur", "Maninagar", "Raikhad", "Rakhial",
        "SAC ISRO Bopal", "SAC ISRO Satellite",
        "SVPI Airport Hansol", "Sardar Vallabhai Patel Stadium"
    ],
    "Latitude": [
        23.11, 22.97, 23.00, 23.03, 23.02,
        23.03, 23.02,
        23.07, 23.04
    ],
    "Longitude": [
        72.58, 72.63, 72.60, 72.58, 72.61,
        72.47, 72.50,
        72.63, 72.57
    ]
})

station_coords


# In[94]:


num_anomalies = df1['Cluster'].value_counts().get(-1, 0)
total_points = len(df1)
percent_anomalies = (num_anomalies / total_points) * 100
print(f"Percentage of anomalies: {percent_anomalies:.2f}%")


# In[99]:


anomalies = df1[df1['Cluster'] == -1]
anomalies['Station_Name'].value_counts()


# In[101]:


df1.index[df1["Cluster"] == -1].month.value_counts()


# # Model Validation

# In[108]:


from sklearn.metrics import silhouette_score #to evaluate clustering quality ranges from -1 to 1; higher is better

mask = df1['Cluster'] != -1
score = silhouette_score(x_scaled[mask], df1.loc[mask, 'Cluster'])
print("Silhouette score:", score)


# In[110]:


from sklearn.metrics import davies_bouldin_score #to evaluate clustering performance (lower values indicate better clustering)

mask = df1['Cluster'] != -1
dbi = davies_bouldin_score(x_scaled[mask], df1.loc[mask, 'Cluster'])

print("Davies-Bouldin Index:", dbi)


# In[112]:


from sklearn.metrics import calinski_harabasz_score #to measure cluster separation (higher values indicate better-defined and well-separated clusters)

mask = df1['Cluster'] != -1
chs = calinski_harabasz_score(x_scaled[mask], df1.loc[mask, 'Cluster'])

print("Calinski-Harabasz Score:", chs)


# In[114]:


noise_ratio = (df1['Cluster'] == -1).mean()
print("Noise Ratio:", noise_ratio)


# In[116]:


print("Stability check:")
for eps in [1.5, 1.6, 1.7]:
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(x_scaled)
    print(eps, (labels == -1).sum())


# In[118]:


df1.groupby(['Station_Name', 'Cluster']).size().reset_index(name='Count')


# In[120]:


anomalies[col].describe()


# # Graphs

# In[123]:


anomalies = df1[df1["Cluster"] == -1].reset_index()

anomaly_month_count = anomalies.groupby(anomalies["Date"].dt.to_period("M")).size().reset_index(name="Anomaly_Count")
anomaly_month_count["Date"] = anomaly_month_count["Date"].astype(str)

fig = px.line(
    anomaly_month_count,
    x="Date",
    y="Anomaly_Count",
    markers=True,
    title="Monthly Environmental Anomalies Over Time",
    labels={"Date": "Month-Year", "Anomaly_Count": "Anomaly Count"}
)

fig.show()


# In[125]:


# as there are 6 features, we can't visualize it, so PCA is used to visualize in 2D and 3D


# In[127]:


from sklearn.decomposition import PCA
import pandas as pd

# Reduce 6D → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)

# Add PCA components into dataframe
df1["PC1"] = X_pca[:, 0]
df1["PC2"] = X_pca[:, 1]

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", pca.explained_variance_ratio_.sum())


# In[129]:


import plotly.express as px

fig = px.scatter(
    df1,
    x="PC1",
    y="PC2",
    color=df1["Cluster"].apply(lambda x: "Anomaly" if x == -1 else "Normal"),
    title="DBSCAN Clustering Visualized Using PCA (All Pollutants)",
    labels={"color": "Point Type"},
    opacity=0.7
)

fig.show()


# In[131]:


pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(x_scaled)

df1["PC3"] = X_pca3[:, 2]

fig = px.scatter_3d(
    df1,
    x="PC1",
    y="PC2",
    z="PC3",
    color=df1["Cluster"].apply(lambda x: "Anomaly" if x == -1 else "Normal"),
    title="3D PCA Visualization of DBSCAN Clusters"
)

fig.show()


# In[133]:


import pandas as pd

# Take anomalies only
anomalies = df1[df1["Cluster"] == -1].reset_index()

# Extract Year and Month
anomalies["Year"] = anomalies["Date"].dt.year
anomalies["Month"] = anomalies["Date"].dt.month_name()

# Order months correctly
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

anomalies["Month"] = pd.Categorical(anomalies["Month"], categories=month_order, ordered=True)

# Count anomalies per Year-Month
heatmap_data = anomalies.groupby(["Year", "Month"]).size().unstack(fill_value=0)

heatmap_data


# In[135]:


import plotly.express as px

fig = px.imshow(
    heatmap_data,
    text_auto=True,
    title="Monthly Environmental Anomaly Heatmap (2023–2025)",
    labels=dict(x="Month", y="Year", color="Anomaly Count"),
)

fig.show()


# In[137]:


station_heatmap = anomalies.groupby(
    ["Station_Name", "Month"]
).size().unstack(fill_value=0)


# In[139]:


fig = px.imshow(
    station_heatmap,
    text_auto=True,
    title="Station-wise Monthly Anomaly Heatmap",
    labels=dict(x="Month", y="Station", color="Anomaly Count")
)

fig.show()


# In[141]:


import pandas as pd

# Count anomalies per station
station_counts = anomalies.groupby("Station_Name").size().reset_index(name="Anomaly_Count")

station_counts


# In[143]:


station_coords = pd.DataFrame({
    "Station_Name": [
        "Chandkheda", "Gyaspur", "Maninagar", "Raikhad", "Rakhial",
        "SAC ISRO Bopal", "SAC ISRO Satellite",
        "SVPI Airport Hansol", "Sardar Vallabhai Patel Stadium"
    ],
    "Latitude": [
        23.11, 22.97, 23.00, 23.03, 23.02,
        23.03, 23.02,
        23.07, 23.04
    ],
    "Longitude": [
        72.58, 72.63, 72.60, 72.58, 72.61,
        72.47, 72.50,
        72.63, 72.57
    ]
})
station_map = station_counts.merge(station_coords, on="Station_Name", how="left")

station_map


# In[145]:


df_map = df1.merge(station_coords, on="Station_Name", how="left")

df_map["Point_Type"] = df_map["Cluster"].apply(
    lambda x: "Anomaly" if x == -1 else "Normal"
)

fig = px.scatter_map(
    df_map,
    lat="Latitude",
    lon="Longitude",
    color="Point_Type",
    hover_name="Station_Name",
    zoom=10,
    title="Normal vs Anomaly Pollution Points Across Ahmedabad Stations",
    opacity=0.6
)

fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[ ]:




