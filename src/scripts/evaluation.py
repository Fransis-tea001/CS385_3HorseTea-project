import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score

def evaluate(kmeans, agg_ward, agg_complete, agg_average):
    df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
    X = df[['recency', 'frequency', 'monetary']]
    X_norm = scaling(X)

    model = [kmeans, agg_ward, agg_complete, agg_average]
    model_score = [calinski_harabasz_score(X_norm, kmeans.labels_),
                   calinski_harabasz_score(X_norm, agg_ward.labels_),
                   calinski_harabasz_score(X_norm, agg_complete.labels_),
                   calinski_harabasz_score(X_norm, agg_average.labels_)]
    
    print("Calinski-Harabasz score k-means : {:.4f}".format(calinski_harabasz_score(X_norm, kmeans.labels_)))
    print("Calinski-Harabasz score Agglomerative (Ward): {:.4f}".format(calinski_harabasz_score(X_norm, agg_ward.labels_)))
    print("Calinski-Harabasz score Agglomerative (Complete): {:.4f}".format(calinski_harabasz_score(X_norm, agg_complete.labels_)))
    print("Calinski-Harabasz score Agglomerative (Average): {:.4f}".format(calinski_harabasz_score(X_norm, agg_average.labels_)))

def scaling(df):
  df_log = np.log1p(df)
  scaler = StandardScaler()
  norm = scaler.fit_transform(df_log)
  return norm