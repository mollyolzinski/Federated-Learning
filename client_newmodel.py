### Run script with two arguments, --model and --partition-id
### Models currently working: LogisticRegression, LinearRegression, LassoCV
### Example: python server.py --model <model-name> --partition-id <integer_to_specify_partition>

import flwr as fl
import utils
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            
        server_round = config["server_round"]
        print(f"Training finished for round {server_round}")

        # Save aggregated_ndarrays
        print(f"Saving round {server_round} parameters...")
        np.savez(f"round-{server_round}-partition-{partition_id}-weights.npz", *parameters)
        
        return utils.get_model_parameters(model), len(X_train), {}
    
    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        server_round = config["server_round"]
        metrics = utils.get_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
            round_number=server_round,
            source='client',
            partition_id=partition_id,
        )

        return metrics['loss'], len(X_test), metrics

if __name__ == "__main__": #run only if the script is being run directly as opposed to being imported from this script
    N_CLIENTS = 10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data CSV file',
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--model",
        type=str,
        #choices=list["LogisticRegression", "LinearRegression", "LassoCV", "SVR"],
        default='LassoCV',
        help="Specifies the model used to fit",
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
    path_data = args.data
    partition_id = args.partition_id
    model_str = args.model
    split_col = args.split_col
    y_col = args.y_col

    #Loads the data and splits into train and test datasets
    (X_client, y_client), _ = utils.get_train_test_data(
        path_csv=path_data,
        partition_id=partition_id,
        split_col=split_col,
        y_col=y_col,
    )
    
    X_train, X_test = X_client[: int(0.8 * len(X_client))], X_client[int(0.8 * len(X_client)) :]
    y_train, y_test = y_client[: int(0.8 * len(y_client))], y_client[int(0.8 * len(y_client)) :]


    #choose model from args input and specify model parameters
    if model_str == 'LogisticRegression':
        model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting    
        )
    elif model_str == 'LinearRegression':
        model = LinearRegression(
            fit_intercept=True,
        )
    elif model_str == 'LassoCV':
        model = LassoCV(
            cv = 5,
            n_alphas = 20,
        )
    # elif model_str == 'SVR':
    #     model = SVR(
            
    #      )
    else:
        raise ValueError(f"Unknown model name: {model_str}")

    utils.set_initial_params(model)

    fl.client.start_client(server_address="0.0.0.0:8080", client=MnistClient().to_client())




