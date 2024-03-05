from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

estimator = {
    "dt": DecisionTreeClassifier,
    "lr": LogisticRegression,
}

estimator_hyper_parameters = {
    "dt": {
        "max_depth": [i for i in range(10)],
        "criterion": ["gini", "entropy"]
    },
    "lr": {
        "C": [ 10 ** (i) for i in range(-4, 3)]
    }
}
