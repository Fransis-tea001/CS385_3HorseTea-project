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

