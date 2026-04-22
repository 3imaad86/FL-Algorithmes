"""ClientApp FedAvg (partitionnement IID ou non-IID configurable)."""

import time

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedavg.task import (
    Net, get_device, load_data, model_size_bytes,
    simulate_comm_delay, test_with_class_accuracies,
)
from fedavg.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    pid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    bs = context.run_config["batch-size"]
    base_epochs = context.run_config["local-epochs"]
    data_hetero = int(context.run_config.get("data-heterogeneity", 0))
    epochs_hetero = int(context.run_config.get("epochs-heterogeneity", 0))
    # Lu depuis run_config (ou surcharge via `flwr run . --run-config ...`).
    partitioning = str(context.run_config.get("partitioning", "noniid"))
    dir_alpha = float(context.run_config.get("dirichlet-alpha", 0.3))
    # Simulation straggler reseau (0 = desactivee, 1 = active).
    straggler_sim = int(context.run_config.get("straggler-sim", 0))
    round_deadline = float(context.run_config.get("round-deadline-s", 0.0))
    # Hetero compute : 3 tiers (weak/medium/strong) via pid % 3 pour simuler
    # des clients edge aux ressources differentes.
    if epochs_hetero:
        tier = pid % 3
        if tier == 0:                               # weak  (IoT sensor)
            epochs = 1
        elif tier == 1:                             # medium (smartphone)
            epochs = 3
        else:                                       # strong (edge server)
            epochs = 6
    else:
        tier = 1
        epochs = base_epochs
    lr = msg.content["config"]["lr"]
    # Le serveur nous envoie le numero de round (sert pour la seed du dropout).
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
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device)
    local_time_s = time.perf_counter() - t0

    # --- Simulation straggler reseau -------------------------------------
    # 1) calcul du delai + tirage du dropout (stable par (pid, round_idx))
    # 2) sleep pour materialiser le delai
    # 3) si dropout ou hors deadline -> on recharge les poids GLOBAUX dans
    #    le modele (= le client n'apporte aucune contribution ce round).
    net_tier, comm_time_s, dropped = 1, 0.0, 0
    if straggler_sim:
        model_mb = model_size_bytes() / (1024.0 * 1024.0)
        net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
        if delay is None:
            dropped = 1                                      # dropout reseau
        else:
            time.sleep(delay)                                # materialise la latence
            comm_time_s = delay
            if round_deadline > 0 and (local_time_s + comm_time_s) > round_deadline:
                dropped = 1                                  # hors deadline
    if dropped:
        model.load_state_dict(global_sd)
    # ---------------------------------------------------------------------

    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "local_time_s": float(local_time_s),
        "partition_id": float(pid),
        "epochs_used": float(epochs),
        "resource_tier": float(tier),
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
