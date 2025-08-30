"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb
from rdagent.utils.fmt import get_xgboost_params


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame) -> xgb.Booster:
    """Define and train the model. Merge feature_select"""
    # 将数据转换为 DMatrix 并指定设备
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Get XGBoost parameters dynamically based on hardware availability
    xgb_params = get_xgboost_params()
    
    params = {
        "learning_rate": 0.5,
        "max_depth": 10,
        **xgb_params,  # This will include tree_method and device parameters
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    num_boost_round = 10

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dvalid, "validation")], verbose_eval=100)
    return model


def predict(model: xgb.Booster, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest).reshape(-1, 1)
    return y_pred