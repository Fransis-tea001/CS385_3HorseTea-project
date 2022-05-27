import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import kneed

def k_means_modelling():
  df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
  X = df[['recency', 'frequency', 'monetary']]
  X_norm = scaling(X)

  kmeans = k_means(X_norm)
  return kmeans

def scaling(df):
  df_log = np.log1p(df)
  scaler = StandardScaler()
  norm = scaler.fit_transform(df_log)
  return norm

def k_means(X):
  sse = {}
  for k in range(1, 21):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(X)
    sse[k] = kmeans.inertia_

  kn = kneed.KneeLocator(
    x=list(sse.keys()), 
    y=list(sse.values()), 
    curve='convex', 
    direction='decreasing')
  
  kmeans = MiniBatchKMeans(n_clusters=kn.knee, random_state=0).fit(X)
  return kmeans