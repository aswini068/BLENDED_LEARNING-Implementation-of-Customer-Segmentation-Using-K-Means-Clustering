# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Import Libraries: Import necessary libraries (pandas, KMeans, StandardScaler, matplotlib).

Load Data: Load the dataset using pandas.read_csv().

Select Features: Extract features: 'Annual Income (k$)' and 'Spending Score (1-100)'.

Scale Data: Standardize the features using StandardScaler.

Determine Optimal K: Use the Elbow Method (plot WCSS) to find the optimal number of clusters.

K-Means Clustering: Perform K-Means clustering with K=5 (optimal clusters).

Assign Cluster Labels: Add the cluster labels to the dataset.

Visualize Clusters: Create a scatter plot of the clusters using Annual Income and Spending Score.
```

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: Aswini M
RegisterNumber: 212223220010

# Import necessary libraries  
import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
  
# Load the Mall Customers dataset  
df = pd.read_csv(r'C:\Users\admin\Downloads\CustomerData.csv')  
  
# Select relevant features for clustering (Annual Income and Spending Score)  
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]  
  
# Scale the data using StandardScaler  
scaler = StandardScaler()  
scaled_features = scaler.fit_transform(features)  
  
# Determine the optimal number of clusters (K) using the Elbow Method  
wcss = []  
for i in range(1, 11):  
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  
 kmeans.fit(scaled_features)  
 wcss.append(kmeans.inertia_)  
  
plt.plot(range(1, 11), wcss)  
plt.title('Elbow Method')  
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS')  
plt.show()  
  
# Perform K-Means clustering with the optimal number of clusters (K=5)  
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)  
kmeans.fit(scaled_features)  
  
# Predict the cluster labels for each customer  
labels = kmeans.labels_  
  
# Add the cluster labels to the original dataset  
df['Cluster'] = labels  
  
# Visualize the clusters using a scatter plot  
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')  
plt.title('Customer Segmentation using K-Means Clustering')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.show()
*/
```

## Output:
![image](https://github.com/user-attachments/assets/5302ad60-7b64-4807-a69c-bf4fda058e2f)
![image](https://github.com/user-attachments/assets/ebc0c6c3-d15c-4e0f-acce-3bc08968fc43)

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
