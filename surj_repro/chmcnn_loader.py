"""C-HMCNN's parser, constraint module and network, for the Surj reproduction.

``./C-HMCNN`` (github.com/EGiunchiglia/C-HMCNN) is imported unmodified through a small
compatibility shim; the R-matrix, ``get_constr_out`` and ``ConstrainedFFNNModel`` are copied
verbatim from ``C-HMCNN/main.py``.

This is the subset of hmc-torch's ``experiments/compare_chmcnn_loader.py`` that the
reproduction uses (the parity check against hmc-torch's own ARFF loader is left out).
"""

from __future__ import annotations

import os
import sys
import types

import networkx as nx
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch import nn

CHMCNN_DIR = "C-HMCNN"

# ---------------------------------------------------------------------------
# Shim: import C-HMCNN's parser unmodified (py2.7 era: keras, nx.to_numpy_matrix)
# ---------------------------------------------------------------------------


def _import_chmcnn():
    keras = types.ModuleType("keras")
    keras.utils = types.SimpleNamespace(to_categorical=lambda i, n: np.eye(n)[i])
    sys.modules.setdefault("keras", keras)
    if not hasattr(nx, "to_numpy_matrix"):
        # their call passes order=nodes, which old networkx accepted and ignored
        nx.to_numpy_matrix = lambda g, nodelist=None, order=None: nx.to_numpy_array(
            g, nodelist=nodelist
        )
    os.environ["DATA_FOLDER"] = CHMCNN_DIR
    sys.path.insert(0, CHMCNN_DIR)
    from utils import datasets as their_datasets  # pylint: disable=import-outside-toplevel
    from utils import parser as their_parser  # pylint: disable=import-outside-toplevel

    sys.path.remove(CHMCNN_DIR)
    return their_parser, their_datasets


THEIR_PARSER, THEIR_DATASETS = _import_chmcnn()

FUNCAT = [n for n in THEIR_DATASETS if n.endswith("_FUN")]
GO = [n for n in THEIR_DATASETS if n.endswith("_GO")]
OTHERS = [n for n in THEIR_DATASETS if n.endswith("_others")]
GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS, "all": FUNCAT + GO + OTHERS}

# README of C-HMCNN, mean over 10 seeds.
PUBLISHED = {
    "cellcycle_FUN": 0.255, "derisi_FUN": 0.195, "eisen_FUN": 0.306, "expr_FUN": 0.302,
    "gasch1_FUN": 0.286, "gasch2_FUN": 0.258, "seq_FUN": 0.292, "spo_FUN": 0.215,
    "cellcycle_GO": 0.413, "derisi_GO": 0.370, "eisen_GO": 0.455, "expr_GO": 0.447,
    "gasch1_GO": 0.436, "gasch2_GO": 0.414, "seq_GO": 0.446, "spo_GO": 0.382,
    "diatoms_others": 0.758, "enron_others": 0.756,
    "imclef07a_others": 0.956, "imclef07d_others": 0.927,
}

# Hyperparameters from C-HMCNN/main.py.
HIDDEN = {
    "FUN": {"cellcycle": 500, "derisi": 500, "eisen": 500, "expr": 1250, "gasch1": 1000,
            "gasch2": 500, "seq": 2000, "spo": 250},
    "GO": {"cellcycle": 1000, "derisi": 500, "eisen": 500, "expr": 4000, "gasch1": 500,
           "gasch2": 500, "seq": 9000, "spo": 500},
    "others": {"diatoms": 2000, "enron": 1000, "imclef07a": 1000, "imclef07d": 1000},
}
LRS = {"FUN": 1e-4, "GO": 1e-4, "others": 1e-5}
EPOCHS = {
    "FUN": {"cellcycle": 106, "derisi": 67, "eisen": 110, "expr": 20, "gasch1": 42,
            "gasch2": 123, "seq": 13, "spo": 115},
    "GO": {"cellcycle": 62, "derisi": 91, "eisen": 123, "expr": 70, "gasch1": 122,
           "gasch2": 177, "seq": 45, "spo": 103},
    "others": {"diatoms": 474, "enron": 133, "imclef07a": 592, "imclef07d": 588},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_theirs(name):
    """(train, val or None, test) as C-HMCNN's arff_data objects."""
    if "others" in name:
        train, test = THEIR_PARSER.initialize_other_dataset(name, THEIR_DATASETS)
        return train, None, test
    return THEIR_PARSER.initialize_dataset(name, THEIR_DATASETS)

def their_r(a):
    """R-matrix exactly as C-HMCNN/main.py builds it (before unsqueeze)."""
    r = np.zeros(a.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(a)
    for i in range(len(a)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0)


def get_constr_out(x, r):
    """C-HMCNN/main.py, verbatim."""
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), r.shape[1], r.shape[1])
    r_batch = r.expand(len(x), r.shape[1], r.shape[1])
    final_out, _ = torch.max(r_batch * c_out.double(), dim=2)
    return final_out



class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h), verbatim from C-HMCNN/main.py."""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, r):
        super().__init__()
        self.nb_layers = hyperparams["num_layers"]
        self.R = r  # pylint: disable=invalid-name
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        self.drop = nn.Dropout(hyperparams["dropout"])
        self.sigmoid = nn.Sigmoid()
        self.f = nn.Tanh() if hyperparams["non_lin"] == "tanh" else nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        return get_constr_out(x, self.R)


def _preprocess(mode, x_tr, x_val, x_te):
    """Return float64 (train, val, test) features for one preprocessing variant."""
    if mode == "none":
        return x_tr, x_val, x_te
    if mode == "chmcnn":  # C-HMCNN/main.py: fit on train+val (train only for _others)
        fit = x_tr if x_val is None else np.concatenate((x_tr, x_val))
        scaler = preprocessing.StandardScaler().fit(fit.astype(float))
        imp = SimpleImputer(missing_values=np.nan, strategy="mean").fit(fit.astype(float))
        tf = lambda x: None if x is None else scaler.transform(imp.transform(x))  # noqa: E731
        return tf(x_tr), tf(x_val), tf(x_te)
    # "train": run_protocol.py's z-score with train stats
    mean = np.nanmean(x_tr, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    xs = [None if x is None else np.where(np.isnan(x), mean, x) for x in (x_tr, x_val, x_te)]
    std = xs[0].std(axis=0)
    std[std == 0] = 1.0
    return tuple(None if x is None else (x - mean) / std for x in xs)
