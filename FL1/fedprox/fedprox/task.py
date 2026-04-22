"""Modele, donnees (IID ou NON-IID Dirichlet), boucles train/test.

Le mode de partitionnement est controle par les parametres `partitioning`
et `alpha` passes a `load_data(...)` / `partition_sizes(...)`.

Valeurs possibles :
  * partitioning = "iid"    -> indices melanges puis coupes en N parts egales
  * partitioning = "noniid" -> Dirichlet(alpha) (alpha petit = plus non-IID)

Ces valeurs viennent de context.run_config ("partitioning" et "dirichlet-alpha")
et sont lues dans client_app.py / server_app.py puis propagees ici.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from .metrics_utils import class_accuracies_from_preds, macro_recall_f1_from_preds

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
VAL_RATIO = 0.2
SEED = int(os.environ.get("FL_SEED", "42"))


class Net(nn.Module):
    """Petit CNN (identique a la base quickstart)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
_trainset = None
# Cache des partitions indexe par (num_parts, partitioning, alpha, seed) pour
# que plusieurs runs avec differents parametres restent coherents.
_parts_cache = {}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_trainset():
    global _trainset
    if _trainset is None:
        _trainset = CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=_transforms)
    return _trainset


# --- PARTITIONNEMENT ------------------------------------------------------
# IID : melange tous les indices au hasard puis coupe en N parts egales.
# Chaque client voit donc (en esperance) toutes les classes en proportion egale.
def _build_iid(num_partitions, seed):
    ds = get_trainset()
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    return [p.tolist() for p in np.array_split(idx, num_partitions)]


# NON-IID : pour chaque classe, tirer une proportion Dirichlet(alpha) entre
# les N clients puis distribuer les exemples selon ces proportions.
# alpha petit -> chaque client specialise sur 1-2 classes.
# alpha grand -> quasi-IID.
def _build_dirichlet(num_partitions, alpha, seed):
    ds = get_trainset()
    targets = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_partitions)]
    for label in np.unique(targets):
        label_idx = np.where(targets == label)[0]
        rng.shuffle(label_idx)
        proportions = rng.dirichlet([alpha] * num_partitions)
        counts = rng.multinomial(len(label_idx), proportions)
        start = 0
        for pid, c in enumerate(counts):
            parts[pid].extend(label_idx[start:start + c].tolist())
            start += c
    # Evite les partitions vides : on pioche 1 exemple chez le plus gros.
    for pid in range(num_partitions):
        if not parts[pid]:
            donor = max(range(num_partitions), key=lambda i: len(parts[i]))
            if len(parts[donor]) > 1:
                parts[pid].append(parts[donor].pop())
    return parts


def build_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    """Construit la liste des indices par client selon le mode demande."""
    mode = str(partitioning).lower()
    if mode == "iid":
        return _build_iid(num_partitions, seed)
    if mode != "noniid":
        raise ValueError(
            f"partitioning={partitioning!r} invalide; valeurs attendues: 'iid' ou 'noniid'"
        )
    if float(alpha) <= 0.0:
        raise ValueError("dirichlet-alpha doit etre > 0 pour un partitionnement noniid")
    return _build_dirichlet(num_partitions, float(alpha), seed)


def get_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    """Version cachee de build_partitions (la cle inclut tous les parametres)."""
    key = (int(num_partitions), str(partitioning).lower(), float(alpha), int(seed))
    if key not in _parts_cache:
        _parts_cache[key] = build_partitions(num_partitions, partitioning, alpha, seed)
    return _parts_cache[key]


def partition_sizes(num_partitions, partitioning="noniid", alpha=0.3):
    """Tailles des partitions (utilise par FedNova pour calculer tau_eff)."""
    return [len(p) for p in get_partitions(num_partitions, partitioning, alpha)]


def load_data(pid, num_partitions, batch_size, data_hetero=0,
              partitioning="noniid", alpha=0.3):
    """Charge (trainloader, valloader) pour le client `pid`.

    Si data_hetero = 1, tronque le TRAIN selon une fraction qui depend du pid
    (keep = 0.2..1.0) pour simuler des tailles differentes entre clients.
    Le VAL reste complet pour que l'evaluation soit comparable.
    """
    ds = get_trainset()
    idx = np.array(get_partitions(num_partitions, partitioning, alpha)[pid])
    np.random.default_rng(SEED + pid).shuffle(idx)
    if len(idx) == 0:
        tr, va = [], []
    elif len(idx) == 1:
        # Evite un trainloader vide sur des partitions tres petites.
        tr = idx.tolist()
        va = idx.tolist()
    else:
        val_size = min(max(1, int(len(idx) * VAL_RATIO)), len(idx) - 1)
        tr, va = idx[:-val_size].tolist(), idx[-val_size:].tolist()
    tr_full = list(tr)
    if int(data_hetero):
        keep = 0.2 + 0.8 * (pid / max(1, num_partitions - 1))
        n_keep = max(1, int(len(tr) * keep))
        tr = tr[:n_keep]
    if not tr:
        # Fallback de securite si le split ou le tronquage ne laisse plus
        # aucun echantillon d'entrainement.
        tr = tr_full[:1] or va[:1]
    return (
        DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=bool(tr)),
        DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False),
    )


def train(net, loader, epochs, lr, device, mu=0.0, global_params=None):
    """Entrainement local multi-epoch. Si mu > 0 et global_params fourni -> FedProx."""
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    tot_loss, tot_ex, steps = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            if mu > 0 and global_params is not None:
                prox = sum(((p - gp) ** 2).sum() for p, gp in zip(net.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            steps += 1
    return tot_loss / max(tot_ex, 1), steps


def fedsgd_update(net, loader, lr, device):
    """FedSGD : UN seul pas de gradient sur UN SEUL mini-batch.

    Le serveur (FedAvg) moyenne ensuite les poids retournes, ce qui revient
    a moyenner les gradients (w_avg = w - lr * mean(g_i)).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)  # SGD pur, pas de momentum
    net.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt.zero_grad()
    loss = crit(net(x), y)
    loss.backward()
    opt.step()
    return loss.item(), y.size(0)


def test(net, loader, device):
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()
    tot_loss, tot_ok, tot_ex = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ok += (out.argmax(1) == y).sum().item()
            tot_ex += bs
    return tot_loss / max(tot_ex, 1), tot_ok / max(tot_ex, 1)


def test_with_class_accuracies(net, loader, device, num_classes=10):
    """Comme test() mais renvoie aussi la liste des accuracies par classe.

    class_accuracies est calcule via sklearn.metrics.confusion_matrix
    (voir metrics_utils.class_accuracies_from_preds).
    """
    net.to(device)
    crit = nn.CrossEntropyLoss().to(device)
    net.eval()
    tot_loss, tot_ex = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y)
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_ex += bs
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    if tot_ex == 0:
        return 0.0, 0.0, [0.0] * num_classes
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    overall_acc = float((y_true == y_pred).sum() / tot_ex)
    class_accs = class_accuracies_from_preds(y_true, y_pred, num_classes=num_classes)
    macro_recall, macro_f1 = macro_recall_f1_from_preds(y_true, y_pred)
    return tot_loss / tot_ex, overall_acc, class_accs, macro_recall, macro_f1


def model_size_bytes():
    """Taille (octets) des parametres -> estimation du volume echange par client."""
    return sum(p.numel() * p.element_size() for p in Net().parameters())


# =========================================================================
# SIMULATION STRAGGLERS : reseau variable + dropouts aleatoires
# =========================================================================
# Motivation : en edge IoT, les clients n'ont pas tous la meme connexion
# (LoRa/2G, LTE, WiFi edge) et peuvent etre injoignables un round (batterie
# faible, coupure reseau). On modelise 3 tiers reseau avec leur taux de dropout.

import random as _rnd

# tier : (bw_mbps, rtt_s, jitter_s, p_drop_par_round)
NET_TIERS = {
    0: (0.5,  0.8,  0.3,  0.15),   # faible (LoRa / 2G)        -> 15% dropout
    1: (5.0,  0.2,  0.05, 0.05),   # moyen  (LTE smartphone)   ->  5% dropout
    2: (50.0, 0.03, 0.01, 0.01),   # fort   (WiFi edge gateway)->  1% dropout
}
# Distribution des tiers dans la flotte (majorite faibles = realiste IoT).
NET_TIER_WEIGHTS = [0.4, 0.4, 0.2]


def network_profile(pid, seed=SEED):
    """Tier reseau STABLE du client (meme pid -> meme tier a tous les rounds).

    Renvoie (tier, bw_mbps, rtt_s, jitter_s, p_drop).
    """
    rng = _rnd.Random(seed + pid)
    tier = rng.choices([0, 1, 2], weights=NET_TIER_WEIGHTS)[0]
    bw, rtt, jitter, pdrop = NET_TIERS[tier]
    return tier, bw, rtt, jitter, pdrop


def simulate_comm_delay(pid, model_mb, round_idx, seed=SEED):
    """Simule la communication d'UN round pour le client `pid`.

    Renvoie (tier, delay_s)  si transfert reussi,
            (tier, None)     si dropout reseau ce round.

    delay = 2*(model_mb/bw) + rtt + |jitter|   (download + upload du modele)
    Le dropout est tire avec p_drop, graine = (seed, pid, round_idx) -> reproductible.
    """
    tier, bw, rtt, jitter, pdrop = network_profile(pid, seed)
    rng = _rnd.Random(seed + pid * 1000 + round_idx)
    if rng.random() < pdrop:
        return tier, None
    delay = 2.0 * (model_mb / bw) + rtt + abs(rng.gauss(0.0, jitter))
    return tier, delay
