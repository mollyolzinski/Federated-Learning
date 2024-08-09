### Run script with two arguments, --model and --min-clients
### Models currently working: LogisticRegression, LinearRegression, LassoCV
### Example: python server.py --model <model-name> --min-clients <integer_no_of_clients>

import flwr as fl
import json
import utils
import numpy as np
import pandas as pd
import argparse
from flwr.common import NDArrays, Scalar
from pathlib import Path
from sklearn.metrics import log_loss, mean_squared_error, r2_score 
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.svm import SVR
from typing import List, Union, Dict, Optional, Tuple

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
    
    #Loads the data and splits into train and test datasets
    _, (X_test, y_test) = utils.get_train_test_data(
        path_csv=path_data,
        split_col=split_col,
        y_col=y_col,
    )

    if isinstance(model, LogisticRegression):
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            
            # Save aggregated_metrics
            print(f"Saving round {server_round} aggregated_metrics...")
            Path(f"round-{server_round}-metrics.json").write_text(json.dumps({
                "loss": loss,
                "accuracy": accuracy,
            }))
            
            return loss, {"accuracy": accuracy}

    if isinstance(model, (LinearRegression, LassoCV)):
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            utils.set_model_params(model, parameters)
            loss = mean_squared_error(y_test, model.predict(X_test))
            accuracy = r2_score(y_test,model.predict(X_test))

            # Save aggregated_metrics
            print(f"Saving round {server_round} aggregated_metrics...")
            Path(f"round-{server_round}-metrics.json").write_text(json.dumps({
                "loss": loss,
                "accuracy": accuracy,
            }))

            return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--model",
        type=str,
        #choices=list["LogisticRegression", "LinearRegression", "LassoCV", "SVR"],
        default='LassoCV',
        help="Specifies the model used to fit",
    )
    parser.add_argument(
       "--min-clients",
        type=int,
        default=1,
        help="Number of clients",
        )
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
    model_str = args.model
    min_clients = args.min_clients
    path_data = args.data
    split_col = args.split_col
    y_col = args.y_col

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
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round, 
        on_evaluate_config_fn=fit_round,
    )  

    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, 
                           config=fl.server.ServerConfig(num_rounds=5))
