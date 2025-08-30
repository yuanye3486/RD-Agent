import pandas as pd
import xgboost as xgb
from rdagent.utils.fmt import get_xgboost_params


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model for both ConfirmedCases and Fatalities."""
    models = {}
    # Get XGBoost parameters dynamically based on hardware availability
    xgb_params = get_xgboost_params()
    
    for target in ["ConfirmedCases", "Fatalities"]:
        dtrain = xgb.DMatrix(X_train, label=y_train[target])
        dvalid = xgb.DMatrix(X_valid, label=y_valid[target])

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": -1,
            **xgb_params,  # This will include tree_method and device parameters
        }
        num_round = 1000

        evallist = [(dtrain, "train"), (dvalid, "eval")]
        models[target] = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=50)

    return models


def predict(models, X):
    """Make predictions for both ConfirmedCases and Fatalities."""
    dtest = xgb.DMatrix(X)
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(dtest)
    return pd.DataFrame(predictions)