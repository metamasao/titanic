import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from datetime import datetime
import argparse
import joblib

import config
from model_dispatcher import estimator
from preprocessing import load_data, concat_data

def train():

    # loading data
    train_data = load_data("train.csv")
    test_data = load_data("test.csv")

    # preprocessing
    concat = concat_data(train_data, test_data)
    age_column = concat.loc[:, ["Age"]]
    fare_column = concat.loc[:, ["Fare"]]

    concat.loc[:, ["Age"]] = age_column.fillna(age_column.mean())
    concat.loc[:, ["Fare"]] = fare_column.fillna(fare_column.mean())

    concat["new_SibSp"] = concat.loc[:, ["SibSp"]].map(lambda x: 1 if x == 0 else 0)
    concat["new_Pclass"] = concat.loc[:, ["Pclass"]].map(lambda x: 1 if x == 3 else 0)

    concat = concat.loc[:, ["Age", "Sex", "Fare", "new_SibSp", "new_Pclass"]]
    concat = pd.get_dummies(concat, drop_first=True, dtype=int)
    
    train_X = concat.iloc[:891]
    test_X = concat.iloc[891:]
    train_y = train_data.loc[:, ["Survived"]]

    # fit
    model = estimator.get("dt")
    model = model(max_depth=4)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    # evaluation of model

    # saving model
    today = datetime.now().date().__str__()
    joblib.dump(model, f"{config.MODEL_DIR}\\{today}model.bin")
    # outputting result
    submission = pd.read_csv("../data/input/gender_submission.csv")
    submission["Survived"] = y_pred
    submission.to_csv(f"../data/output/{today}submission.csv", index=False)

train()