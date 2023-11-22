from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import argparse

import tensorflow as tf
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

import os

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print(f"datas from clients : {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    sum_loss = sum(losses) / len(losses)
    sum_acc = sum(accuracies) / sum(examples)
    print(f"total loss : {sum_loss}")
    print(f"total accuracy : {sum_acc}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum_acc}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--server_port',
    type=str,
    default="8083"
)
parser.add_argument(
    '--num_clients',
    type=int,
    default=2
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=10
)
parser.add_argument(
    "--tb_port",
    type=str,
    default=6006
)
args=parser.parse_args()

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=args.num_clients,
    min_evaluate_clients=args.num_clients,
    min_available_clients=args.num_clients
)

fl.common.logger.configure(identifier="evc_test", filename="flwr_logs/fl_log.txt")
fl.common.Status("client.py", "helloworld")

# Start Flower server
data = fl.server.start_server(
    server_address=f"0.0.0.0:{args.server_port}",
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
)

# print(data.losses_distributed[0][0])
# print(data.losses_distributed[0][1])
# print(data.metrics_distributed["accuracy"][0][0])
# print(data.metrics_distributed["accuracy"][0][1])

df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy'])

for i in range(len(data.losses_distributed)):
    new_data = {
        'Round' : data.losses_distributed[i][0],
        'Loss' : data.losses_distributed[i][1],
        'Accuracy' : data.metrics_distributed['accuracy'][i][1]
    }
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

print(df)

def visualize_tensorboard(df):
    writer = SummaryWriter()  # SummaryWriter 생성

    # TensorBoard에 기록할 데이터를 데이터프레임에서 추출
    rounds = df['Round']
    losses = df['Loss']
    accuracies = df['Accuracy']

    # TensorBoard에 Loss와 Accuracy 기록
    for i in range(len(rounds)):
        writer.add_scalar('Loss/Per_Round', losses[i], rounds[i])
        writer.add_scalar('Accuracy/Per_Round', accuracies[i], rounds[i])

    writer.close()  # SummaryWriter 닫기

# 함수 실행
visualize_tensorboard(df)

# 로그 디렉토리 경로
log_directory = './runs'

# TensorBoard 실행
os.system(f"tensorboard --logdir {log_directory} --port {args.tb_port} --bind_all")