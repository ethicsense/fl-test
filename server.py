from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import argparse


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--server_port',
    type=str,
    default="8083"
)
parser.add_argument(
    '--num_clients',
    type=int,
    default=1
)
args=parser.parse_args()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=args.num_clients,
    min_evaluate_clients=args.num_clients,
    min_available_clients=args.num_clients
)

fl.common.logger.configure(identifier="evcFLtest", filename="log.txt")

# Start Flower server
fl.server.start_server(
    server_address=f"0.0.0.0:{args.server_port}",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
