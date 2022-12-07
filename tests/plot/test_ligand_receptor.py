import itertools

import numpy as np

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

LIGAND_RECEPTOR_ROOT = ROOT / "ligand_receptor"
LIGAND_RECEPTOR_FIGS = FIGS / "ligand_receptor"


def test_ligand_receptor_scatter(dense_sample, image_comparer):
    np.random.seed(0)
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(dense_sample, contacts, list(itertools.product("GHKL", "ABCI")))
    lr_stat_scores = mb.stat.ligand_receptor_score(dense_sample, contacts, lr_scores)
    mb.plot.ligand_receptor_scatter(lr_scores, lr_stat_scores, show=False)

    save_and_compare_images = image_comparer(LIGAND_RECEPTOR_ROOT, LIGAND_RECEPTOR_FIGS, tol=15)
    save_and_compare_images("scatter")
