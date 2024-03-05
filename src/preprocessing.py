import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import DATA_INPUT_DIR

def load_data(filename):
    data = pd.read_csv(os.path.join(DATA_INPUT_DIR, filename))
    return data

def concat_data(train_X, test_X):
    return pd.concat([train_X, test_X], sort=False)

def normalize_SS(train_X, test_X):
    ss = StandardScaler()
    ss.fit(train_X)

    std_train_X = ss.transform(train_X)
    std_test_X = ss.transform(test_X)
    return std_train_X, std_test_X

def fillna_with_model(model):
    pass
