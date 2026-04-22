"""ServerApp FedAvg (partitionnement IID ou non-IID configurable).

Le mode de partitionnement (`partitioning`, `dirichlet-alpha`) est lu dans
run_config cote CLIENTS. Cote serveur, on l'affiche juste pour tracabilite.
"""

import logging
import os
import shutil
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fedavg.metrics_utils import (
    RESULTS_DIR, ensure_dir, log_participation, log_round, log_summary,
    reset_files, resolve_dst_results_dir, rounds_to_convergence, rounds_to_target,
)
from fedavg.task import Net, model_size_bytes

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Silence les logs internes de Flower ([ROUND 1/1], banniere strategie, etc.)
    # pour que seuls nos prints [round N] et [early-stop] apparaissent.
    logging.getLogger("flwr").setLevel(logging.WARNING)
    reset_files()
    cfg = context.run_config
    num_rounds = int(cfg["num-server-rounds"])
    frac_eval = float(cfg["fraction-evaluate"])
    lr = float(cfg["learning-rate"])
    num_clients = int(cfg.get("num-clients", 10))
    partitioning = str(cfg.get("partitioning", "noniid"))
    dir_alpha = float(cfg.get("dirichlet-alpha", 0.3))
    es_patience = int(cfg.get("early-stopping-patience", 0))
    es_min_delta = float(cfg.get("early-stopping-min-delta", 0.001))
    straggler_sim = int(cfg.get("straggler-sim", 0))
    round_deadline_s = float(cfg.get("round-deadline-s", 0.0))

    extra = f" alpha={dir_alpha}" if partitioning.lower() != "iid" else ""
    strag = f" straggler-sim=1 deadline={round_deadline_s}s" if straggler_sim else ""
    print(f"[config] algo=FedAvg partitioning={partitioning}{extra} "
          f"num_rounds={num_rounds} lr={lr}{strag}")

    model_mb = model_size_bytes() / (1024.0 * 1024.0)
    t_start = time.perf_counter()
    state = {
        "round": 0,
        "accs_history": [],
        "t_last": t_start,
        "last_train_times": [],
        "n_train_clients_round": 0,
        "participation": {},
        "mean_epochs_used": 0.0,
        "mean_resource_tier": 0.0,
        "clients_detail": [],
        "best_acc": 0.0,
        "no_improve": 0,
        "early_stop": False,
    }

    def agg_train(records, wk):
        recs = list(records)
        m = aggregate_metricrecords(recs, wk)
        state["n_train_clients_round"] = len(recs)
        times, epochs_list, tier_list = [], [], []
        clients_detail = []
        n_dropped = 0
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            pid = int(mr.get("partition_id", -1))
            if pid >= 0:
                state["participation"][pid] = state["participation"].get(pid, 0) + 1
            times.append(float(mr.get("local_time_s", 0.0)))
            epochs_list.append(float(mr.get("epochs_used", 0.0)))
            tier_list.append(float(mr.get("resource_tier", 1.0)))
            is_dropped = int(float(mr.get("dropped", 0.0)) >= 0.5)
            n_dropped += is_dropped
            clients_detail.append({
                "pid": pid,
                "n": int(mr.get("num-examples", 0)),
                "epochs": float(mr.get("epochs_used", 0.0)),
                "tier": float(mr.get("resource_tier", 1.0)),
                "time": float(mr.get("local_time_s", 0.0)),
                "net_tier": int(mr.get("net_tier", 1)),
                "comm_time": float(mr.get("comm_time_s", 0.0)),
                "dropped": is_dropped,
            })
        state["last_train_times"] = times
        state["clients_detail"] = clients_detail
        state["mean_epochs_used"] = sum(epochs_list) / max(len(epochs_list), 1)
        state["mean_resource_tier"] = sum(tier_list) / max(len(tier_list), 1)
        state["n_dropped_round"] = n_dropped
        return m

    def agg_eval(records, wk):
        recs = list(records)
        m = aggregate_metricrecords(recs, wk)
        state["round"] += 1
        r = state["round"]

        global_acc = float(m.get("accuracy", 0.0))
        global_loss = float(m.get("loss", 0.0))
        macro_recall = float(m.get("macro_recall", 0.0))
        macro_f1 = float(m.get("macro_f1", 0.0))
        state["accs_history"].append(global_acc)

        n_eval_clients = len(recs)
        n_train_clients = int(state.get("n_train_clients_round", n_eval_clients))
        comm_mb = 2.0 * n_train_clients * model_mb

        now = time.perf_counter()
        round_time_s = now - state["t_last"]
        state["t_last"] = now
        tt = state["last_train_times"] or [0.0]
        mean_ct = float(sum(tt) / len(tt))
        max_ct = float(max(tt))

        client_accs = []
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            client_accs.append(float(mr.get("accuracy", 0.0)))

        class_accs = list(m.get("class_accuracies", [0.0] * 10))

        log_round(r, global_acc, global_loss, comm_mb, macro_recall, macro_f1, client_accs, class_accs,
                  round_time_s=round_time_s, mean_client_time_s=mean_ct,
                  max_client_time_s=max_ct,
                  mean_epochs_used=state["mean_epochs_used"],
                  mean_resource_tier=state["mean_resource_tier"])
        # Detail par client participant a ce round (visibilite edge IoT).
        tier_names = {0: "weak", 1: "medium", 2: "strong"}
        net_names = {0: "lora", 1: "lte", 2: "wifi"}
        print(f"[round {r}] clients participants ({n_train_clients}):")
        for c in sorted(state["clients_detail"], key=lambda x: x["pid"]):
            tname = tier_names.get(int(c["tier"]), "?")
            nname = net_names.get(int(c.get("net_tier", 1)), "?")
            flag = " DROP" if c.get("dropped", 0) else ""
            print(f"  pid={c['pid']:>2}  n={c['n']:>5}  E={c['epochs']:.0f}  "
                  f"tier={tname:<6}  net={nname:<4}  t={c['time']:.2f}s  "
                  f"comm={c.get('comm_time', 0.0):.2f}s{flag}")
        if straggler_sim:
            print(f"[round {r}] stragglers dropped={state.get('n_dropped_round', 0)}/{n_train_clients}")
        print(f"[round {r}] acc={global_acc:.3f} loss={global_loss:.3f} "
              f"recall={macro_recall:.3f} f1={macro_f1:.3f} "
              f"comm={comm_mb:.2f}MB n={n_train_clients} "
              f"round={round_time_s:.1f}s mean_ct={mean_ct:.2f}s "
              f"E={state['mean_epochs_used']:.2f} tier={state['mean_resource_tier']:.2f}")

        if es_patience > 0:
            if global_acc > state["best_acc"] + es_min_delta:
                state["best_acc"] = global_acc
                state["no_improve"] = 0
            else:
                state["no_improve"] += 1
                if state["no_improve"] >= es_patience:
                    state["early_stop"] = True
                    print(f"[early-stop] convergence detectee a r={r} "
                          f"(best_acc={state['best_acc']:.3f}, "
                          f"patience={es_patience}, min_delta={es_min_delta})")
        return m

    strategy = FedAvg(
        fraction_evaluate=frac_eval,
        train_metrics_aggr_fn=agg_train,
        evaluate_metrics_aggr_fn=agg_eval,
    )

    # Boucle manuelle : 1 round a la fois pour pouvoir s'arreter tot si
    # early stopping declenche (state["early_stop"] est leve dans agg_eval).
    arrays = ArrayRecord(Net().state_dict())
    result = None
    for round_idx in range(1, num_rounds + 1):
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            # "round" est lu cote client pour grainer le dropout reseau.
            train_config=ConfigRecord({"lr": lr, "round": round_idx}),
            num_rounds=1,
        )
        arrays = result.arrays
        if state["early_stop"]:
            break

    total_time = time.perf_counter() - t_start
    log_summary(total_time, state["accs_history"], state["participation"], num_clients=num_clients)
    log_participation(state["participation"], num_clients=num_clients)

    rtc = rounds_to_convergence(state["accs_history"], ratio=0.9)
    r50 = rounds_to_target(state["accs_history"], 0.5)
    r70 = rounds_to_target(state["accs_history"], 0.7)
    r90 = rounds_to_target(state["accs_history"], 0.9)
    print(f"[done] total_time={total_time:.1f}s rtc90%={rtc} "
          f"r50={r50} r70={r70} r90={r90} | partitioning={partitioning}{extra}")
    torch.save(result.arrays.to_torch_state_dict(), os.path.join(ensure_dir(), "final_model.pt"))
    print(f"[done] CSV -> {RESULTS_DIR}")
    dst = resolve_dst_results_dir("fedavg")
    if os.path.abspath(dst) != os.path.abspath(RESULTS_DIR):
        try:
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(RESULTS_DIR):
                shutil.copy2(os.path.join(RESULTS_DIR, fn), os.path.join(dst, fn))
            print(f"[done] CSV copies dans {dst}")
        except Exception as e:
            print(f"[done] WARN copie CSV echouee: {e}")
