"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb
from rdagent.utils.fmt import get_xgboost_params


def fit(X_train, y_train, X_valid, y_valid):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Get XGBoost parameters dynamically based on hardware availability
    xgb_params = get_xgboost_params()
    
    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "num_class": 10,
        "nthread": -1,
        **xgb_params,  # This will include tree_method and device parameters
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    model = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

    return model


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    return model.predict(dtest).astype(int)