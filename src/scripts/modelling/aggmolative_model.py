import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score 
import kneed

def agglomerative_modelling(lk):
  df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
  X = df[['recency', 'frequency', 'monetary']]
  X_norm = scaling(X)

  agg = agglomerative(X_norm, lk)
  X_agg = df.assign(cluster=agg.labels_)
  scoring_log(X_norm, agg, lk)
  return agg, X_agg

def scaling(df):
  df_log = np.log1p(df)
  scaler = StandardScaler()
  norm = scaler.fit_transform(df_log)
  return norm

def agglomerative(X, lk):
  sse = {}
  for k in range(1, 21):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(X)
    sse[k] = kmeans.inertia_

  kn = kneed.KneeLocator(
    x=list(sse.keys()), 
    y=list(sse.values()), 
    curve='convex', 
    direction='decreasing')
  
  agg = AgglomerativeClustering(n_clusters=kn.knee, linkage=lk).fit(X)
  return agg

def scoring_log(X, agg, lk):
  with open('..\CS385_3HorseTea-project\logs\Model_out_log.txt', 'a+') as f:
    f.seek(0)
    data = f.read(100)

    if len(data) > 0 :
        f.write("\n")

    f.write("Model : [Agglomerative ({})]\n".format(lk))
    f.write("Silhouette score = {:.4f}\n".format(silhouette_score(X, agg.labels_)))
    f.write("Calinski-Harabasz score = {:.4f}\n".format(calinski_harabasz_score(X, agg.labels_)))
    f.write("Davies-Bouldin score = {:.4f}\n".format(davies_bouldin_score(X, agg.labels_)))
    f.write("\n")
    f.write("------------------------------\n")