import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score

def evaluate(model, model_result, model_names):
    df = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
    X = df[['recency', 'frequency', 'monetary']]
    X_norm = scaling(X)

    model_score = [calinski_harabasz_score(X_norm, model[i].labels_) for i in range(len(model))]
    best_model = select_best_model(model, model_names, model_score, model_result)

    scoring_log(X_norm, best_model)
    save_outputs(best_model[2])

def scaling(df):
  df_log = np.log1p(df)
  scaler = StandardScaler()
  norm = scaler.fit_transform(df_log)
  return norm

def select_best_model(model, m_names, m_score, m_result):
  maxpos = m_score.index(max(m_score))
  return model[maxpos], m_names[maxpos], m_result[maxpos]

def save_outputs(df_cluster):
    df_ori = pd.read_csv("../CS385_3HorseTea-project/data/raw data/data_raw.csv")
    df_clean = pd.read_csv("../CS385_3HorseTea-project/data/processed data/Customer_clean.csv")
    df_cluster = df_cluster[['Customer_ID', 'cluster']]

    df_merge = df_ori.merge(df_clean, on='Customer_ID')
    df_merge = df_merge.merge(df_cluster, on='Customer_ID')
    df_merge.to_csv('..\CS385_3HorseTea-project\outputs\Customer_segmentation.csv', index=False)

def scoring_log(X, model):
  with open('..\CS385_3HorseTea-project\logs\Model_out_log.txt', 'a+') as f:
    f.seek(0)
    data = f.read(100)

    if len(data) > 0 :
        f.write("\n")

    f.write("choose model : {} with Calinski-Harabasz score = {:.4f}\n".format(model[1], calinski_harabasz_score(X, model[0].labels_)))