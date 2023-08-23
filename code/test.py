import numpy as np 
import pandas as pd
import os 
from sklearn.preprocessing import LabelEncoder

cat_cols = [
    'region', 'city', 'parent_category_name', 'category_name',
    'param_1', 'param_2', 'param_3', 'image_top_1'
]

#train = pd.read_csv('./_.csv')

train = pd.read_csv('E:\\kaggle-test\\train.csv', usecols=cat_cols+['deal_probability'])
test = pd.read_csv('E:\\kaggle-test\\test.csv', usecols=cat_cols)

print(cat_cols)
# fill in missing
train['image_top_1'] = train['image_top_1'].astype(str)
test['image_top_1'] = test['image_top_1'].astype(str)
train.fillna('', inplace=True)
test.fillna('', inplace=True)

for col in cat_cols:
    le = LabelEncoder()
    le.fit(np.concatenate([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

class BetaEncoder(object):        

   def __init__(self, group):

      self.group = group
      self.stats = None

   # get counts from df
   def fit(self, df, target_col):
      self.prior_mean = np.mean(df[target_col])
      stats = df[[target_col, self.group]].groupby(self.group)
      stats = stats.agg(['sum', 'count'])[target_col]    
      stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
      stats.reset_index(level=0, inplace=True)           
      self.stats = stats

   # extract posterior statistics
   def transform(self, df, stat_type, N_min=1):
      print(N_min)
      df_stats = pd.merge(df[[self.group]], self.stats, how='left')
      n = df_stats['n'].copy()
      N = df_stats['N'].copy()
      print("----------the N" ,N)
      print('----------the n',n)
      # fill in missing
      nan_indexs = np.isnan(n)
      n[nan_indexs] = self.prior_mean
      N[nan_indexs] = 1.0

      # prior parameters
      N_prior = np.maximum(N_min-N, 0)
      print('----------N_prior',N_prior)
      alpha_prior = self.prior_mean*N_prior
      beta_prior = (1-self.prior_mean)*N_prior
      print('----------alpha_prior',alpha_prior)
      # posterior parameters
      alpha = alpha_prior + n
      print('----------alpha',alpha)
      beta =  beta_prior + N-n
      print('----------alpha',beta)
      # calculate statistics
      if stat_type=='mean':
         num = alpha
         dem = alpha+beta
                  
      elif stat_type=='mode':
         num = alpha-1
         dem = alpha+beta-2
         
      elif stat_type=='median':
         num = alpha-1/3
         dem = alpha+beta-2/3

      elif stat_type=='var':
         num = alpha*beta
         dem = (alpha+beta)**2*(alpha+beta+1)
                  
      elif stat_type=='skewness':
         num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
         dem = (alpha+beta+2)*np.sqrt(alpha*beta)

      elif stat_type=='kurtosis':
         num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
         dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
         
      # replace missing
      value = num/dem
      value[np.isnan(value)] = np.nanmedian(value)
      return value

N_min = 1000
feature_cols = []    

# encode variables
for c in cat_cols:
   print(c)
   be = BetaEncoder(c)
   be.fit(train, 'deal_probability')

   # mean
   feature_name = f'{c}_mean'
   train[feature_name] = be.transform(train, 'mean', N_min)
   test[feature_name]  = be.transform(test,  'mean', N_min)
   feature_cols.append(feature_name)