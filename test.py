import flwr as fl
import utils
import numpy as np
import pandas as pd
import argparse
import utils
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, mean_squared_error, r2_score 
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import List, Union, Dict, Optional, Tuple

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data CSV file',
    )
    parser.add_argument(
        "--split-col",
        type=str,
        default='3_splits',
        help="Name of column used to split the data into train partitions and test set",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default='age',
        help="Name of output variable column",
    )
    args = parser.parse_args()
    
    (X_train, y_train), (X_test, y_test) = utils.get_train_test_data(
        path_csv=args.data,
        partition_id=None,
        split_col=args.split_col,
        y_col=args.y_col,
    )
    
    model = LassoCV(
        cv = 5,
        n_alphas = 20,
    )
    
    model.fit(X_train, y_train)
    
    loss = mean_squared_error(y_test, model.predict(X_test))
    accuracy = r2_score(y_test,model.predict(X_test))
    
    print(loss, accuracy)