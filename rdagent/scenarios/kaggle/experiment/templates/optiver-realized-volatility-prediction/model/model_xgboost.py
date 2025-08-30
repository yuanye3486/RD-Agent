import pandas as pd
import xgboost as xgb
from rdagent.utils.fmt import get_xgboost_params


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Get XGBoost parameters dynamically based on hardware availability
    xgb_params = get_xgboost_params()
    
    # Parameters for regression
    params = {
        "objective": "reg:squarederror",  # Use squared error for regression
        "nthread": -1,
        **xgb_params,  # This will include tree_method and device parameters
    }
    num_round = 200

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
    return y_pred.reshape(-1, 1)