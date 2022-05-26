import numpy as np
import pandas as pd

from datetime import date
from datetime import datetime

def preprocessing():
  df = pd.read_csv("../CS385_3HorseTea-project/data/raw data/data_raw.csv")
  df_clean = df_cleansing(df)
  df_wrag = data_wagling(df_clean)
  df_wrag = data_processing(df_wrag)
  df_wrag.to_csv('..\CS385_3HorseTea-project\data\processed data\Customer_clean.csv', index=False)
  return df_wrag

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
  return df_clean

def data_wagling(df):
  df_r = calculate_recency(df)
  df_rf = df_r.merge(calculate_frequency(df), on='Customer_ID')
  df_rfm = df_rf.merge(calculate_monetary(df), on='Customer_ID')

  return df_rfm

def calculate_recency(df):
  NOW = date.today()
  for i in range(len(df['date'])):
    df.loc[i, 'recency'] = (NOW - df.loc[i, 'date']).days
  df = df[['Customer_ID', 'recency']]
  return df

def calculate_frequency(df):
  df = df.drop_duplicates().groupby(by=['Customer_ID'], as_index=False)['date'].count()
  df.columns = ['Customer_ID', 'frequency']
  return df

def calculate_monetary(df):
  df = df.groupby('Customer_ID', as_index=False).sum()
  df['monetary'] = df['Product_Price'] * df['Quantity']
  df = df[['Customer_ID', 'monetary']]
  return df

def data_processing(df):
  df_copy = df.copy()
  df_segt = rfm_score(df_copy)
  df_segt = customer_segmentation(df_segt)
  return df_segt

def rfm_score(df):
  quantiles = df.quantile(q=[.2, .4, .6, .8]).to_dict()
  df['r_score'] = df['recency'].apply(RScore, args=('recency',quantiles,))
  df['f_score'] = df['frequency'].apply(FMScore, args=('frequency',quantiles,))
  df['m_score'] = df['monetary'].apply(FMScore, args=('monetary',quantiles,))
  df['RFMScore'] = df.r_score.map(str) + df.f_score.map(str) + df.m_score.map(str)
  return df

def RScore(x,p,d):
  if x <= d[p][0.2]:
    return 5
  elif x <= d[p][0.4]:
    return 4
  elif x <= d[p][0.6]: 
    return 3
  elif x <= d[p][0.8]: 
   return 2
  else:
   return 1
    
def FMScore(x,p,d):
  if x <= d[p][0.2]:
    return 1
  elif x <= d[p][0.4]:
    return 2
  elif x <= d[p][0.6]: 
    return 3
  elif x <= d[p][0.8]: 
   return 4
  else:
   return 5
  
def customer_segmentation(df):
  segt_map = {
    r'[4-5][3-5][4-5]': 'Champions',
    r'[2-4][3-5][4-5]': 'Loyal Customers',
    r'[3-4][3-5][1-4]': 'Need Attention',
    r'[4-5][4-5][1-3]': 'Small basket',
    r'[3-5][1-3][1-3]': 'Potential Loyalists',
    r'[4-5]11': 'New Customers',
    r'[3-4]11': 'Promising',
    r'[3-5][1-2][4-5]': 'Long time Big Buy',
    r'[1-2][1-5][1-5]': 'At Risk',
    r'[2-3][1-3][1-3]': 'Hibernating',
    r'[1-3][1-2][1-2]': 'About to Sleep',
    r'111': 'Lost'
  }

  df['Segment'] = df['RFMScore'].replace(segt_map, regex=True)
  return df