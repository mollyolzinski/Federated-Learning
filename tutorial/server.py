import flwr as fl
import utils
import numpy as np
from flwr.common import NDArrays, Scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, Tuple

from flwr_datasets import FederatedDataset

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    dataset = fds.load_split("test").with_format("numpy")
    X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(server_address="localhost:5678", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))