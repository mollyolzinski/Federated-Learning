### Run script with two arguments, --model and --partition-id
### Models currently working: LogisticRegression, LinearRegression, LassoCV
### Example: python server.py --model <model-name> --partition-id <integer_to_specify_partition>

import flwr as fl
import utils
import argparse
import warnings
import pandas as pd
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import Dict
import matplotlib.pyplot as plt

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(model), len(X_train), {}
    
    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        if isinstance(model, LogisticRegression):
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
        if isinstance(model, (LinearRegression, LassoCV)):
            loss = mean_squared_error(y_test, model.predict(X_test))
            accuracy = r2_score(y_test,model.predict(X_test))
        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__": #run only if the script is being run directly as opposed to being imported from this script
    N_CLIENTS = 10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--model",
        type=str,
        #choices=list["LogisticRegression", "LinearRegression", "LassoCV", "SVR"],
        required=True,
        help="Specifies the model used to fit",
    )
    args = parser.parse_args()
    partition_id = args.partition_id
    model_str = args.model

#Loads the data and splits into train and test datasets
    dataset_full:pd.DataFrame = pd.read_csv("hbn_fs_data_split.csv")
    dataset_full["sex"] = dataset_full["sex"].map({"M": 1, "F": 2})
    dataset_full = dataset_full.drop (columns = ['subject_id', 
                                                 'scan_site_id', 'ehq_total', 'commercial_use', 
                                                 'full_pheno', 'expert_qc_score', 'xgb_qc_score', 
                                                 'xgb_qsiprep_qc_score', 'dl_qc_score', 'site_variant',
                                                 'age_category', 'stratify_col'])
    dataset_client:pd.DataFrame = dataset_full.loc[dataset_full['3_splits'] != partition_id]
    y = dataset_client["age"]
    X = dataset_client.drop(columns = ["age"])
    #fds = FederatedDataset(dataset="mnist", partitioners={"train": N_CLIENTS})
    #dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    #X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]


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




