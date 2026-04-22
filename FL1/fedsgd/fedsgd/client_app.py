"""ClientApp FedSGD (partitionnement IID ou non-IID configurable).

Difference avec FedAvg : UN seul pas SGD sur UN mini-batch (fedsgd_update).
"""

import time

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedsgd.task import (
    Net, fedsgd_update, get_device, load_data, model_size_bytes,
    simulate_comm_delay, test_with_class_accuracies,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    pid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    bs = context.run_config["batch-size"]
    data_hetero = int(context.run_config.get("data-heterogeneity", 0))
    # Note : FedSGD = 1 seul pas SGD -> epochs-heterogeneity n'a pas de sens ici.
    partitioning = str(context.run_config.get("partitioning", "noniid"))
    dir_alpha = float(context.run_config.get("dirichlet-alpha", 0.3))
    straggler_sim = int(context.run_config.get("straggler-sim", 0))
    round_deadline = float(context.run_config.get("round-deadline-s", 0.0))
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))

    model = Net()
    global_sd = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(pid, num_parts, bs,
                               data_hetero=data_hetero,
                               partitioning=partitioning, alpha=dir_alpha)
    t0 = time.perf_counter()
    train_loss, _ = fedsgd_update(model, trainloader, lr, device)
    local_time_s = time.perf_counter() - t0

    # Simulation straggler reseau (voir commentaire fedavg/client_app.py).
    net_tier, comm_time_s, dropped = 1, 0.0, 0
    if straggler_sim:
        model_mb = model_size_bytes() / (1024.0 * 1024.0)
        net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
        if delay is None:
            dropped = 1
        else:
            time.sleep(delay)
            comm_time_s = delay
            if round_deadline > 0 and (local_time_s + comm_time_s) > round_deadline:
                dropped = 1
    if dropped:
        model.load_state_dict(global_sd)

    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "local_time_s": float(local_time_s),
        "partition_id": float(pid),
        # FedSGD = 1 pas SGD, pas de notion d'epochs ni de tier ; on loggue
        # des constantes pour garder un schema CSV homogene avec les autres algos.
        "epochs_used": 1.0,
        "resource_tier": 1.0,
        "net_tier": float(net_tier),
        "comm_time_s": float(comm_time_s),
        "dropped": float(dropped),
    })
    return Message(
        content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    pid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    bs = context.run_config["batch-size"]
    partitioning = str(context.run_config.get("partitioning", "noniid"))
    dir_alpha = float(context.run_config.get("dirichlet-alpha", 0.3))

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    _, valloader = load_data(pid, num_parts, bs,
                             partitioning=partitioning, alpha=dir_alpha)
    loss, accuracy, class_accuracies, macro_recall, macro_f1 = test_with_class_accuracies(
        model, valloader, device
    )

    metrics = MetricRecord({
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_accuracies": [float(a) for a in class_accuracies],
        "num-examples": len(valloader.dataset),
        "partition_id": float(pid),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
