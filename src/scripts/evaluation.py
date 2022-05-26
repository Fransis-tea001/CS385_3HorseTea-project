import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score 

def evaluate(kmeans, agg_ward, agg_complete, agg_average):
    df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
    X = df[['recency', 'frequency', 'monetary']]
    X_norm = scaling(X)

    model = [kmeans, agg_ward, agg_complete, agg_average]
    model_names = ['k-Means', 'Agglomerative (Ward)', 'Agglomerative (Complete)', 'Agglomerative (Average)']
    model_score = [calinski_harabasz_score(X_norm, kmeans.labels_),
                   calinski_harabasz_score(X_norm, agg_ward.labels_),
                   calinski_harabasz_score(X_norm, agg_complete.labels_),
                   calinski_harabasz_score(X_norm, agg_average.labels_)]
    
    best_model = select_best_model(model, model_names, model_score)

    print("choose model : {} with Calinski-Harabasz score = {:.4f}".format(best_model[1], calinski_harabasz_score(X_norm, best_model[0].labels_)))

    save_log(model, model_names, best_model, X_norm)
    save_outputs(df, X_norm, best_model[0])

def scaling(df):
  df_log = np.log1p(df)
  scaler = StandardScaler()
  norm = scaler.fit_transform(df_log)
  return norm

def select_best_model(model, m_names, m_score):
  maxpos = m_score.index(max(m_score))
  return model[maxpos], m_names[maxpos]

def save_log(model, m_names, best_model, X):
    with open('..\CS385_3HorseTea-project\logs\Model_out_log.txt', 'w') as f:
        for i in range(len(model)):
            f.write("Model : [{}]\n".format(m_names[i]))
            f.write("Calinski-Harabasz score = {:.4f}\n".format(silhouette_score(X, model[i].labels_)))
            f.write("Silhouette score = {:.4f}\n".format(calinski_harabasz_score(X, model[i].labels_)))
            f.write("Davies-Bouldin score = {:.4f}\n".format(davies_bouldin_score(X, model[i].labels_)))
            f.write("\n")
            f.write("------------------------------\n")
        f.write("choose : {} model\n".format(best_model[1]))

def save_outputs(X, X_norm, model):
    best_model = model.fit(X_norm)
    X_best_model = X.assign(cluster=best_model.labels_)
    X_best_model.to_csv('..\CS385_3HorseTea-project\outputs\Customer_seg_result.csv', index=False)