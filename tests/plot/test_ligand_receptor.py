import numpy as np

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

LIGAND_RECEPTOR_ROOT = ROOT / "ligand_receptor"
LIGAND_RECEPTOR_FIGS = FIGS / "ligand_receptor"


def test_ligand_receptor_scatter(sample_with_expression, image_comparer):
    np.random.seed(0)
    contacts = mb.calc.cell_contact(sample_with_expression, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(sample_with_expression, contacts, [("A", "B"), ("A", "C")])
    lr_stat_scores = mb.stat.ligand_receptor_score(sample_with_expression, contacts, lr_scores)
    mb.plot.ligand_receptor_scatter(lr_scores, lr_stat_scores, show=False)

    save_and_compare_images = image_comparer(LIGAND_RECEPTOR_ROOT, LIGAND_RECEPTOR_FIGS, tol=15)
    save_and_compare_images("scatter")
