import numpy as np
import pandas as pd

from datetime import date
from datetime import datetime

def preprocessing():
  df = pd.read_csv("../CS385_3HorseTea-project/data/raw data/data_raw.csv")
  df_clean = df_cleansing(df)
  return df_clean

def date_sep(df):
  for d in range(len(df['Order_Date'])):
    df.loc[d, 'date'] = df.loc[d, 'Order_Date'].date()
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
  return df.drop(['Order_Date'],axis=1)

def cleaning_outlier(df):
  df_num = [i for i in df.dtypes.index if df.dtypes[i] == 'int64']
  df_num_out = [i for i in df_num if len(find_outliers_IQR(df[i])) > 0]

  if len(df_num_out) == 0:
    return df
  else:
    for i in df_num_out:
      tenth_q = df[i].quantile(0.10)
      ninth_q = df[i].quantile(0.90)
      df[i] = np.where(df[i] < tenth_q, tenth_q, df[i])
      df[i] = np.where(df[i] > ninth_q, ninth_q, df[i])
    return df

def find_outliers_IQR(df):
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3-Q1
  outliers = df[((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR)))]
  return outliers

def df_cleansing(df):
  df_clean = df.copy()

  # Drop non-meaning columns
  df_clean = df_clean.drop(['Customer_Name', 'ZIP', 'Address', 'Tel'], axis=1)

  # Convert Order_ID and Customer_ID to String
  df_clean['Order_ID'], df_clean['Customer_ID'] = df_clean['Order_ID'].astype('str'), df_clean['Customer_ID'].astype('str')
  df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'])

  # Separate Date and time
  df_clean = date_sep(df_clean)

  # Find Outlier and Remove the outlier using the Quantile based floring and capping technique.
  df_clean = cleaning_outlier(df_clean)

  df_clean = df_clean[df_clean['date'] < date.today()]
  df_clean.to_csv('..\CS385_3HorseTea-project\data\processed data\Customer_clean.csv', index=False)
  return df_clean