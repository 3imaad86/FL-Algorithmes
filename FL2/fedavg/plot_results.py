"""Trace les courbes a partir des CSV dans results/.

Figures produites dans results/ :
  - accuracy.png         : accuracy globale par round
  - loss.png             : loss globale par round
  - recall_f1.png        : macro recall et macro F1 par round
  - comm_cost.png        : communication cost cumule par round
  - convergence_rounds.png : rounds-to-target + rtc90
  - fairness_jfi.png     : Jain's Fairness Index (clients + classes)
  - worst_case.png       : worst-case accuracy (clients + classes)
  - per_class.png        : accuracy par classe (10 courbes)
  - round_time.png       : temps par round + moyenne/max compute client
  - participation.png    : nb de selections par client (participation fairness)
"""

import csv
import os

import matplotlib.pyplot as plt

RESULTS = "results"
GLOBAL_CSV = os.path.join(RESULTS, "metrics_global.csv")
PER_CLASS_CSV = os.path.join(RESULTS, "metrics_per_class.csv")
PARTICIPATION_CSV = os.path.join(RESULTS, "metrics_participation.csv")
SUMMARY_CSV = os.path.join(RESULTS, "metrics_summary.csv")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"-> {path}")


def plot_accuracy(rows):
    rs = [int(r["round"]) for r in rows]
    accs = [float(r["global_accuracy"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, accs, "-o")
    ax.set_xlabel("round"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy globale par round (convergence)")
    ax.grid(alpha=.3)
    save(fig, "accuracy.png")


def plot_loss(rows):
    rs = [int(r["round"]) for r in rows]
    losses = [float(r["global_loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, losses, "-o", color="tab:red")
    ax.set_xlabel("round"); ax.set_ylabel("loss")
    ax.set_title("Loss globale par round")
    ax.grid(alpha=.3)
    save(fig, "loss.png")


def plot_recall_f1(rows):
    rs = [int(r["round"]) for r in rows]
    recalls = [float(r["macro_recall"]) for r in rows]
    f1s = [float(r["macro_f1"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, recalls, "-o", label="macro recall")
    ax.plot(rs, f1s, "-s", label="macro F1")
    ax.set_xlabel("round"); ax.set_ylabel("score")
    ax.set_title("Macro Recall et Macro F1 par round")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "recall_f1.png")


def plot_comm_cost(rows):
    rs = [int(r["round"]) for r in rows]
    per_round = [float(r["comm_cost_mb"]) for r in rows]
    cumul = []
    acc = 0.0
    for v in per_round:
        acc += v
        cumul.append(acc)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, cumul, "-o", color="tab:purple")
    ax.set_xlabel("round"); ax.set_ylabel("MB cumules")
    ax.set_title("Communication cost cumule par round")
    ax.grid(alpha=.3)
    save(fig, "comm_cost.png")


def plot_convergence_summary(rows):
    if not rows:
        return
    row = rows[0]
    labels = ["rtc90", "r50", "r70", "r90"]
    values = [
        float(row["rtc90"]) if row.get("rtc90", "") != "" else 0.0,
        float(row["rounds_to_50"]) if row.get("rounds_to_50", "") != "" else 0.0,
        float(row["rounds_to_70"]) if row.get("rounds_to_70", "") != "" else 0.0,
        float(row["rounds_to_90"]) if row.get("rounds_to_90", "") != "" else 0.0,
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="tab:orange")
    ax.set_xlabel("metric"); ax.set_ylabel("rounds")
    ax.set_title("Metriques de convergence")
    ax.grid(alpha=.3, axis="y")
    save(fig, "convergence_rounds.png")


def plot_fairness_jfi(rows):
    rs = [int(r["round"]) for r in rows]
    jfi_c = [float(r["jfi_clients"]) for r in rows]
    jfi_k = [float(r["jfi_classes"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, jfi_c, "-o", label="JFI clients")
    ax.plot(rs, jfi_k, "-s", label="JFI classes")
    ax.set_xlabel("round"); ax.set_ylabel("Jain's Fairness Index")
    ax.set_ylim(0, 1.05)
    ax.set_title("Jain's Fairness Index par round")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "fairness_jfi.png")


def plot_worst_case(rows):
    rs = [int(r["round"]) for r in rows]
    worst_c = [float(r["worst_client_acc"]) for r in rows]
    worst_k = [float(r["worst_class_acc"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, worst_c, "-o", label="worst client")
    ax.plot(rs, worst_k, "-s", label="worst class")
    ax.set_xlabel("round"); ax.set_ylabel("accuracy")
    ax.set_title("Worst-case accuracy (clients + classes) par round")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "worst_case.png")


def plot_round_time(rows):
    if "round_time_s" not in rows[0]:
        return
    rs = [int(r["round"]) for r in rows]
    rt = [float(r["round_time_s"]) for r in rows]
    mean_ct = [float(r["mean_client_time_s"]) for r in rows]
    max_ct = [float(r["max_client_time_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, rt, "-o", label="round wall-clock")
    ax.plot(rs, mean_ct, "-s", label="mean client compute")
    ax.plot(rs, max_ct, "-^", label="max client compute")
    ax.set_xlabel("round"); ax.set_ylabel("seconds")
    ax.set_title("Temps par round (wall-clock + compute client)")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "round_time.png")


def plot_participation(rows):
    if not rows:
        return
    cids = [int(r["client_id"]) for r in rows]
    counts = [int(r["times_selected"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cids, counts, color="tab:green")
    ax.set_xlabel("client_id"); ax.set_ylabel("nb selections")
    ax.set_title("Participation par client (fairness de selection)")
    ax.set_xticks(cids)
    ax.grid(alpha=.3, axis="y")
    save(fig, "participation.png")


def plot_per_class(rows):
    if not rows:
        return
    rs = [int(r["round"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in range(10):
        key = f"class_{c}"
        ys = [float(r[key]) for r in rows]
        ax.plot(rs, ys, "-o", label=key, markersize=3)
    ax.set_xlabel("round"); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy par classe par round")
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=.3)
    save(fig, "per_class.png")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    g = read_csv(GLOBAL_CSV)
    pc = read_csv(PER_CLASS_CSV)
    part = read_csv(PARTICIPATION_CSV)
    summary = read_csv(SUMMARY_CSV)
    if not g:
        print(f"[warn] {GLOBAL_CSV} vide/absent")
    else:
        plot_accuracy(g)
        plot_loss(g)
        plot_recall_f1(g)
        plot_comm_cost(g)
        plot_fairness_jfi(g)
        plot_worst_case(g)
        plot_round_time(g)
    if summary:
        plot_convergence_summary(summary)
    if not pc:
        print(f"[warn] {PER_CLASS_CSV} vide/absent")
    else:
        plot_per_class(pc)
    if part:
        plot_participation(part)
    plt.show()


if __name__ == "__main__":
    main()
