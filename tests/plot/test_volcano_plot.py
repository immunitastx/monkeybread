import random

import pandas as pd
import scanpy as sc

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

VOLCANO_PLOT_ROOT = ROOT / "volcano_plot"
VOLCANO_PLOT_FIGS = FIGS / "volcano_plot"


def test_volcano_plot_cell_type(dense_sample, image_comparer):
    random.seed(0)
    sc.tl.rank_genes_groups(dense_sample, groupby="cell_type")
    mb.plot.volcano_plot(dense_sample, group="ct1", title="cell_type_rank_genes_groups", show=False)

    save_and_compare_images = image_comparer(VOLCANO_PLOT_ROOT, VOLCANO_PLOT_FIGS, tol=15)
    save_and_compare_images("cell_type")


def test_volcano_plot_contact(dense_sample, image_comparer):
    random.seed(0)
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    dense_sample.obs["contact"] = pd.Categorical(
        ["contact" if c in contacts else "no_contact" for c in dense_sample.obs.index]
    )
    sc.tl.rank_genes_groups(dense_sample, groupby="contact")
    mb.plot.volcano_plot(dense_sample, group="contact", title="contact_rank_genes_groups", show=False)

    save_and_compare_images = image_comparer(VOLCANO_PLOT_ROOT, VOLCANO_PLOT_FIGS, tol=15)
    save_and_compare_images("contact")
