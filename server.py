### Run script with two arguments, --model and --min-clients
### Models currently working: LogisticRegression, LinearRegression, LassoCV
### Example: python server.py --model <model-name> --min-clients <integer_no_of_clients>

import flwr as fl
import utils
import numpy as np
import pandas as pd
import argparse
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss, mean_squared_error, r2_score 
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import List, Union, Dict, Optional, Tuple

from flwr_datasets import FederatedDataset

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    
def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    #fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    #dataset = fds.load_split("test").with_format("numpy")
    #X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"] #change according to line
    dataset_full:pd.DataFrame = pd.read_csv("hbn_fs_data_split.csv")
    dataset_full["sex"] = dataset_full["sex"].map({"M": 1, "F": 2})
    dataset_full = dataset_full.drop (columns = ['subject_id', 
                                                 'scan_site_id', 'ehq_total', 'commercial_use', 
                                                 'full_pheno', 'expert_qc_score', 'xgb_qc_score', 
                                                 'xgb_qsiprep_qc_score', 'dl_qc_score', 'site_variant', 
                                                 'age_category', 'stratify_col'])
    dataset_test:pd.DataFrame = dataset_full.loc[dataset_full['3_splits'] == -1]
    y_test = dataset_test["age"]
    X_test = dataset_test.drop(columns = ["age"])

    if isinstance(model, LogisticRegression):
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, {"accuracy": accuracy}

        return evaluate
    if isinstance(model, (LinearRegression, LassoCV)):
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            utils.set_model_params(model, parameters)
            loss = mean_squared_error(y_test, model.predict(X_test))
            accuracy = r2_score(y_test,model.predict(X_test))
            return loss, {"accuracy": accuracy}

        return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--model",
        type=str,
        #choices=list["LogisticRegression", "LinearRegression", "LassoCV", "SVR"],
        required=True,
        help="Specifies the model used to fit",
    )
    parser.add_argument(
       "--min-clients",
        type=int,
        required=True,
        help="Number of clients",
        )
    args = parser.parse_args()
    model_str = args.model
    min_clients = args.min_clients

  #choose model from args input and specify model parameters
    if model_str == 'LogisticRegression':
        model = LogisticRegression()
    elif model_str == 'LinearRegression':
        model = LinearRegression()
    elif model_str == 'LassoCV':
        model = LassoCV()
    # elif model_str == 'SVR':
    #     model = SVR(    
    #      )
    else:
        raise ValueError(f"Unknown model name: {model_str}")

    utils.set_initial_params(model)
    strategy = SaveModelStrategy(
        min_available_clients=3,
        min_fit_clients=2,
        min_evaluate_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round, 
    )  

    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, 
                           config=fl.server.ServerConfig(num_rounds=5))
