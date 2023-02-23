import sys
import os
import warnings
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import scipy#pip install scipy implicit catboost seaborn matplotlib plotly xgboost lightgbm
import implicit
import bisect
import sklearn.metrics as m
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sns.set_style('darkgrid')

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

LOCAL_DATA_PATH = './context_data/'
SPLIT_SEED = 42
DATA_FILE = 'competition_data_final_pqt'
TARGET_FILE = 'competition_target_pqt'

data = pq.read_table(f'{LOCAL_DATA_PATH}/{DATA_FILE}')
targets = pq.read_table(f'{LOCAL_DATA_PATH}/{TARGET_FILE}')

data_agg = data.select(['user_id', 'url_host', 'request_cnt']).group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")])

url_set = set(data_agg.select(['url_host']).to_pandas()['url_host'])
url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
usr_set = set(data_agg.select(['user_id']).to_pandas()['user_id'])
usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}

values = np.array(data_agg.select(['request_cnt_sum']).to_pandas()['request_cnt_sum'])
rows = np.array(data_agg.select(['user_id']).to_pandas()['user_id'].map(usr_dict))
cols = np.array(data_agg.select(['url_host']).to_pandas()['url_host'].map(url_dict))
mat = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))

als = implicit.approximate_als.FaissAlternatingLeastSquares(factors=50,
iterations=30,
use_gpu=False,
calculate_training_loss=False,
regularization=0.1)
als.fit(mat)
u_factors = als.model.user_factors
d_factors = als.model.item_factors

inv_usr_map = {v: k for k, v in usr_dict.items()}
usr_emb = pd.DataFrame(u_factors)
usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
df = targets.to_pandas().merge(usr_emb, how='inner', on=['user_id'])

def age_bucket(x):
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)

df = df[df['age'] != 'NA']
df = df.dropna()
df['age'] = df['age'].map(age_bucket)
sns.histplot(df['age'], bins=7)

x_train, x_test, y_train, y_test = train_test_split(df.drop(['user_id', 'age', 'is_male'], axis=1),
                                                    df['age'],
                                                    test_size=0.33,
                                                    random_state=SPLIT_SEED)




ns= StandardScaler()
xn_train=x_train
xn_test=x_test
yn_train=y_train
yn_test=y_test

# Initialize Random Forest Classifier and CatBoostClassifier
#rf = RandomForestClassifier(n_estimators=50, random_state=42)
xgb = XGBClassifier(n_estimators=200, random_state=42)
lgbm = LGBMClassifier(n_estimators=200, random_state=42)
#cb = CatBoostClassifier(iterations=50, random_state=42)

# Create ensemble model with voting
ensemble_model = VotingClassifier(
    estimators=[
        #('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm),
        #('cb', cb)
    ],
    voting='soft'
)

# Fit ensemble model on training set
ensemble_model.fit(xn_train, yn_train)
print("TRENING END")


print(f'GINI по возрасту {2 * m.roc_auc_score(yn_test, ensemble_model.predict_proba(xn_test)[:,1]) - 1:2.3f}')

df = targets.to_pandas().merge(usr_emb, how = 'inner', on = ['user_id'])
df = df[df['is_male'] != 'NA']
df = df.dropna()
df['is_male'] = df['is_male'].map(int)
df['is_male'].value_counts()
x_train, x_test, y_train, y_test = train_test_split(\
    df.drop(['user_id', 'age', 'is_male'], axis = 1), df['is_male'], test_size = 0.33, random_state = SPLIT_SEED)
xn_train=x_train
xn_test=x_test
yn_train=y_train
yn_test=y_test
#rf = RandomForestClassifier(n_estimators=50, random_state=42)
xgb = XGBClassifier(n_estimators=200, random_state=42)
lgbm = LGBMClassifier(n_estimators=200, random_state=42)
#cb = CatBoostClassifier(iterations=50, random_state=42)

# Create ensemble model with voting
ensemble_model = VotingClassifier(
    estimators=[
        #('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm),
        #('cb', cb)
    ],
    voting='soft'
)

# Fit ensemble model on training set
ensemble_model.fit(xn_train, yn_train)
print(f'GINI по полу {2 * m.roc_auc_score(yn_test, ensemble_model.predict_proba(xn_test)[:,1]) - 1:2.3f}')