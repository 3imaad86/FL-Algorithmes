"""Metriques Federated Learning (efficacite + fairness).

Produit 4 fichiers CSV dans results/ :
  - metrics_global.csv        : 1 ligne/round (efficacite + fairness + temps)
  - metrics_per_class.csv     : 1 ligne/round (accuracy par classe)
  - metrics_summary.csv       : 1 ligne finale (total time, rounds-to-target, participation)
  - metrics_participation.csv : 1 ligne/client (times_selected)

Dependances : numpy, csv, os, sklearn.metrics.confusion_matrix.
"""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

# OU sont ecrits les CSV ?
# 1) Si FL_RESULTS_DIR est fournie (run_seeds.py la set), on l'utilise.
# 2) Sinon on ecrit a cote de __file__ (Flower copie le projet dans ~/.flwr/apps/...
#    donc les CSV seront dans ce cache ; le print [done] affiche le chemin exact).
RESULTS_DIR = os.environ.get(
    "FL_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results"),
)
NUM_CLASSES = 10

GLOBAL_CSV = os.path.join(RESULTS_DIR, "metrics_global.csv")
PER_CLASS_CSV = os.path.join(RESULTS_DIR, "metrics_per_class.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
PARTICIPATION_CSV = os.path.join(RESULTS_DIR, "metrics_participation.csv")

GLOBAL_HEADER = [
    "round", "global_accuracy", "global_loss", "comm_cost_mb",
    "macro_recall", "macro_f1",
    "jfi_clients", "worst_client_acc", "acc_variance_clients",
    "jfi_classes", "worst_class_acc", "acc_variance_classes", "min_max_class_gap",
    "round_time_s", "mean_client_time_s", "max_client_time_s",
    "mean_epochs_used", "mean_resource_tier",
]
PER_CLASS_HEADER = ["round"] + [f"class_{i}" for i in range(NUM_CLASSES)]
SUMMARY_HEADER = [
    "total_time_s", "rounds_to_50", "rounds_to_70", "rounds_to_90",
    "participation_jfi", "worst_participation", "best_participation",
]
PARTICIPATION_HEADER = ["client_id", "times_selected"]


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def resolve_dst_results_dir(project_dir_name):
    """Retourne le dossier ou copier les CSV hors du cache Flower.

    `project_dir_name` = nom du DOSSIER projet (pas forcement le package Python ;
    ex: FedSGD a pour dossier 'fedSGD_noniid' et package 'fedavg_noniid').

    Priorite :
      1) FL_RESULTS_DIR (env var) s'il est fourni.
      2) Chercher un dossier projet nomme <project_dir_name> contenant un
         pyproject.toml, en explorant CWD, ses parents, et leurs enfants.
         Utile sur Kaggle/Colab ou l'utilisateur lance `flwr run <chemin>`
         depuis un autre CWD.
      3) Fallback : <CWD>/results.
    """
    if "FL_RESULTS_DIR" in os.environ:
        return os.environ["FL_RESULTS_DIR"]
    cwd = Path(os.getcwd()).resolve()

    def _matches(p):
        return p.is_dir() and p.name == project_dir_name and (p / "pyproject.toml").exists()

    # Cas A : on est deja dans le dossier projet.
    if _matches(cwd):
        return str(cwd / "results")

    # Cas B : le dossier projet est un enfant direct du CWD.
    try:
        for child in cwd.iterdir():
            if _matches(child):
                return str(child / "results")
    except OSError:
        pass

    # Cas C : on est dans un dossier frere (ex: Kaggle, CWD=.../fedavg_iid,
    # on cherche .../fedprox_noniid). Remonter et explorer les enfants de chaque parent.
    for parent in cwd.parents:
        if _matches(parent):
            return str(parent / "results")
        try:
            for sibling in parent.iterdir():
                if _matches(sibling):
                    return str(sibling / "results")
        except OSError:
            continue

    # Fallback : CWD.
    return str(cwd / "results")


def reset_files():
    ensure_dir()
    for p in (GLOBAL_CSV, PER_CLASS_CSV, SUMMARY_CSV, PARTICIPATION_CSV):
        if os.path.exists(p):
            os.remove(p)


def _append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _overwrite(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


# --- Helpers metriques ----------------------------------------------------

def jains_fairness_index(values):
    """JFI = (sum x)^2 / (n * sum x^2). 1.0 = parfaitement equitable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    s = arr.sum()
    sq = float((arr * arr).sum())
    if sq == 0.0:
        return 0.0
    return float((s * s) / (arr.size * sq))


def class_accuracies_from_preds(y_true, y_pred, num_classes=NUM_CLASSES):
    """Cote CLIENT : [acc_class_0, ..., acc_class_{C-1}] via matrice de confusion."""
    labels = list(range(num_classes))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return [0.0] * num_classes
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accs = []
    for c in labels:
        tot = int(cm[c].sum())
        accs.append(float(cm[c, c] / tot) if tot > 0 else 0.0)
    return accs


def macro_recall_f1_from_preds(y_true, y_pred):
    """Cote CLIENT : recall macro et F1 macro."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(macro_recall), float(macro_f1)


def rounds_to_convergence(accuracies, ratio=0.9):
    """Premier round ou acc >= ratio * max(acc). Renvoie None si liste vide."""
    accs = list(accuracies)
    if not accs:
        return None
    threshold = ratio * max(accs)
    for i, a in enumerate(accs, start=1):
        if a >= threshold:
            return i
    return None


def rounds_to_target(accuracies, target):
    """Premier round ou acc >= target (seuil absolu). Renvoie None si non atteint."""
    for i, a in enumerate(accuracies, start=1):
        if a >= target:
            return i
    return None


# --- Logging CSV ----------------------------------------------------------

def log_round(server_round, global_accuracy, global_loss, comm_cost_mb,
              macro_recall, macro_f1,
              client_accuracies, class_accuracies,
              round_time_s=0.0, mean_client_time_s=0.0, max_client_time_s=0.0,
              mean_epochs_used=0.0, mean_resource_tier=0.0):
    """Cote SERVEUR : calcule fairness + temps et ecrit les 2 CSV par round."""
    ensure_dir()

    client_accs = [float(a) for a in (client_accuracies or [])]
    class_accs = [float(a) for a in (class_accuracies or [])]
    if len(class_accs) < NUM_CLASSES:
        class_accs = class_accs + [0.0] * (NUM_CLASSES - len(class_accs))
    else:
        class_accs = class_accs[:NUM_CLASSES]

    # Fairness Niveau 1 : entre clients
    jfi_c = jains_fairness_index(client_accs)
    worst_c = float(min(client_accs)) if client_accs else 0.0
    var_c = float(np.var(client_accs)) if client_accs else 0.0

    # Fairness Niveau 2 : entre classes
    jfi_k = jains_fairness_index(class_accs)
    worst_k = float(min(class_accs))
    var_k = float(np.var(class_accs))
    gap_k = float(max(class_accs) - min(class_accs))

    _append(
        GLOBAL_CSV, GLOBAL_HEADER,
        [server_round, float(global_accuracy), float(global_loss), float(comm_cost_mb),
         float(macro_recall), float(macro_f1),
         jfi_c, worst_c, var_c, jfi_k, worst_k, var_k, gap_k,
         float(round_time_s), float(mean_client_time_s), float(max_client_time_s),
         float(mean_epochs_used), float(mean_resource_tier)],
    )
    _append(
        PER_CLASS_CSV, PER_CLASS_HEADER,
        [server_round] + class_accs,
    )


def log_summary(total_time_s, accs_history, participation_counts,
                targets=(0.5, 0.7, 0.9), num_clients=None):
    """Ecrit metrics_summary.csv (1 ligne, overwrite)."""
    ensure_dir()
    rt = [rounds_to_target(accs_history, t) for t in targets]
    if isinstance(participation_counts, dict):
        if num_clients is not None:
            counts = [int(participation_counts.get(cid, 0)) for cid in range(int(num_clients))]
        else:
            counts = list(participation_counts.values())
    else:
        counts = list(participation_counts) if participation_counts else []
    p_jfi = jains_fairness_index(counts) if counts else 0.0
    worst_p = int(min(counts)) if counts else 0
    best_p = int(max(counts)) if counts else 0
    _overwrite(
        SUMMARY_CSV, SUMMARY_HEADER,
        [[float(total_time_s),
          rt[0] if rt[0] is not None else "",
          rt[1] if rt[1] is not None else "",
          rt[2] if rt[2] is not None else "",
          p_jfi, worst_p, best_p]],
    )


def log_participation(participation_counts, num_clients=None):
    """Ecrit metrics_participation.csv (1 ligne/client, overwrite)."""
    ensure_dir()
    if num_clients is not None:
        rows = [[cid, int(participation_counts.get(cid, 0))] for cid in range(int(num_clients))]
    else:
        rows = [[int(cid), int(n)] for cid, n in sorted(participation_counts.items())]
    _overwrite(PARTICIPATION_CSV, PARTICIPATION_HEADER, rows)


# --- Utilitaire Flower ----------------------------------------------------

def extract_server_round(msg):
    """Lit le numero de round depuis le Message Flower."""
    cfg = msg.content.get("config")
    if cfg is not None:
        sr = cfg.get("server-round")
        if sr is not None:
            return int(sr)
    md = getattr(msg, "metadata", None)
    if md is not None:
        gid = getattr(md, "group_id", None)
        if gid:
            try:
                return int(gid)
            except (TypeError, ValueError):
                pass
    return -1
