#!/usr/bin/env python

from collections import defaultdict
from frozendict import frozendict
import pickle

from pyscenic.cli.utils import load_adjacencies, load_exp_matrix
from pyscenic.utils import modules_from_adjacencies


adj_csv = "outputs/GRNBoost2/combined/adj.csv"
loom = "outputs/prepare_scenic/filtered_scenic.loom"

adj_mod_dat = "outputs/find_modules/adj_mod.dat"


adjacencies = load_adjacencies(adj_csv)


ex_mtx = load_exp_matrix(
    loom,
    True,  # transpose
    False,  # sparse loading is disabled here for now
    "CellID",  # cell_id_attribute
    "Gene",  # gene_attribute
)

modules = modules_from_adjacencies(
    adjacencies,
    ex_mtx,
    thresholds = [0.75, 0.90],  # The first method to create the TF-modules based on the best targets for each transcription factor (default: 0.75 0.90).
    top_n_targets = [50],  # The second method is to select the top targets for a given TF. (default: 50)
    top_n_regulators = [5, 10, 50],  # The alternative way to create the TF-modules is to select the best regulators for each gene. (default: 5 10 50)
    min_genes = 20,  # The minimum number of genes in a module (default: 20).
    rho_mask_dropouts = False,
    keep_only_activating = False,
)

with open(adj_mod_dat, "wb") as f:
    pickle.dump(modules, f)
