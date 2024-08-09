from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, r2_score 
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR

from flwr.common import NDArrays

def get_train_test_data(path_csv: Path, split_col: str, y_col: str, partition_id: int = None, 
                        keep_features: list[str] = None):
    """Returns ((X_train, y_train), (X_test, y_test))."""

    # load dataframe
    dataset_full: pd.DataFrame = pd.read_csv(path_csv)

    # fix sex
    dataset_full["sex"] = dataset_full["sex"].map({"M": 1, "F": 2})
    
    #create a list of columns specifying splits to drop
    extra_columns_with_splits = [col for col in dataset_full.columns if 'splits' in col and col != split_col]

    # drop unwanted columns
    dataset_full = dataset_full.drop (columns = ['subject_id', 
                                                 'scan_site_id', 'ehq_total', 'commercial_use', 
                                                 'full_pheno', 'expert_qc_score', 'xgb_qc_score', 
                                                 'xgb_qsiprep_qc_score', 'dl_qc_score', 'site_variant',
                                                 'age_category', 'stratify_col'] + extra_columns_with_splits)
    
    if keep_features is not None:
        dataset_full = dataset_full.loc[:, keep_features]

    if partition_id is None:
        train_idx = dataset_full[split_col] != -1
    else:
        train_idx = dataset_full[split_col] == partition_id
    
    dataset_train: pd.DataFrame = dataset_full.loc[train_idx]
    y_train = dataset_train[y_col]
    X_train = dataset_train.drop(columns = [y_col])
    
    dataset_test: pd.DataFrame = dataset_full.loc[dataset_full[split_col] == -1]
    y_test = dataset_test[y_col]
    X_test = dataset_test.drop(columns = [y_col])

    return (X_train, y_train), (X_test, y_test)

def get_metrics(model, X_test, y_test, round_number, source, partition_id=None) -> dict:
    metrics = {
        'round_number': round_number,
        'source': source,
        'partition_id': partition_id,
    }
    if isinstance(model, LogisticRegression):
        metrics.update({
            'loss': log_loss(y_test, model.predict_proba(X_test)),
            'accuracy': model.score(X_test, y_test),
        })
    elif isinstance(model, (LinearRegression, LassoCV)):
        y_pred = model.predict(X_test)
        metrics.update({
            'loss': mean_squared_error(y_test, y_pred),
            'accuracy': r2_score(y_test, y_pred),
        })
    else:
        raise RuntimeError(f'Unsupported model: {model}')

    return metrics

def get_model_parameters(model) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if isinstance (model,(LogisticRegression, LinearRegression, LassoCV)):
        if model.fit_intercept:
            params = [
                model.coef_,
                model.intercept_,
            ]
        else:
            params = [
                model.coef_,
            ]
    return params

def set_model_params(model, params: np.ndarray):
    """
    Sets the parameters of a scikit-learn model. Supports LogisticRegression, LinearRegression, and LassoCV.
    
    Parameters:
    - model: The model to be updated (must be one of LogisticRegression, LinearRegression, or LassoCV).
    - params: An array of parameters to set. For LogisticRegression and LinearRegression,
              params[0] should be the coefficients and params[1] should be the intercept (if applicable).
              For LassoCV, params[0] should be the coefficients.
    
    Returns:
    - model: The updated model with the new parameters.
    """
    if isinstance(model, (LogisticRegression, LinearRegression, LassoCV)):
        model.coef_ = params[0]
        if model.fit_intercept:
            model.intercept_ = params[1]
    else:
        raise ValueError("Unsupported model type. Please use LogisticRegression, LinearRegression, or LassoCV.")
    
    return model


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    if isinstance(model, (LogisticRegression)):
        n_classes = 10  # adapt to # of age bin classes
        n_features = 131  # adapt to number of features in the dataset
        model.classes_ = np.array([i for i in range(10)])

        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))

    elif isinstance(model, (LinearRegression, LassoCV)):
        n_features = 131 # adapt to number of features in the dataset

        model.coef_ = np.zeros(n_features)
        if model.fit_intercept:
            model.intercept_ = 0
    