import os.path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score 
import kneed

def k_means_modelling():
  df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
  X = df[['recency', 'frequency', 'monetary']]
  X_norm = scaling(X)
  
  kmeans = k_means(X_norm)
  X_kmeans = df.assign(cluster=kmeans.labels_)
  scoring_log(X_norm, kmeans)
  return kmeans, X_kmeans

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

def scoring_log(X, k_means):
  filesize = os.path.getsize("..\CS385_3HorseTea-project\logs\Model_out_log.txt")

  if filesize == 0:
    with open('..\CS385_3HorseTea-project\logs\Model_out_log.txt', 'a+') as f:
      f.seek(0)
      data = f.read(100)

      if len(data) > 0 :
        f.write("\n")

      f.write("Model : [k-Means]\n")
      f.write("Silhouette score = {:.4f}\n".format(silhouette_score(X, k_means.labels_)))
      f.write("Calinski-Harabasz score = {:.4f}\n".format(calinski_harabasz_score(X, k_means.labels_)))
      f.write("Davies-Bouldin score = {:.4f}\n".format(davies_bouldin_score(X, k_means.labels_)))
      f.write("\n")
      f.write("------------------------------\n")
  else:
    f = open("..\CS385_3HorseTea-project\logs\Model_out_log.txt","r+")
    f.truncate(0)
    f.seek(0)
    data = f.read(100)

    if len(data) > 0 :
      f.write("\n")

    f.write("Model : [k-Means]\n")
    f.write("Silhouette score = {:.4f}\n".format(silhouette_score(X, k_means.labels_)))
    f.write("Calinski-Harabasz score = {:.4f}\n".format(calinski_harabasz_score(X, k_means.labels_)))
    f.write("Davies-Bouldin score = {:.4f}\n".format(davies_bouldin_score(X, k_means.labels_)))
    f.write("\n")
    f.write("------------------------------\n")
    f.close()