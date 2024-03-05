import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve

sns.set_theme()

def plot_validation_curve(
    model, 
    X, 
    y, 
    params, 
    param_name, 
    param_range, 
    ax,
    log=False, 
    cv=5
):
    train_scores, test_scores = validation_curve(
        estimator=model(random_state=0),
        X=X,
        y=y,
        cv=5,
        param_name=param_name,
        param_range=param_range
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    ax.plot(params, train_mean, marker="^", markersize=3, color="blue", label="training")
    ax.fill_between(params, train_mean + train_std, train_mean - train_std, color="blue", alpha=0.2)

    ax.plot(params, test_mean, marker="s", markersize=3, color="red", label="test")
    ax.fill_between(params, test_mean + test_std, test_mean - test_std, color="red", alpha=0.2)

    if (log):
        ax.xscale("log")

    ax.legend(loc="upper left")

def plot_validation_curves(
    model, 
    X, 
    y, 
    params, 
    param_name, 
    param_range, 
    ax,
    log=False, 
    cv=5
):
    pass

