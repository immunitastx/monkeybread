import itertools

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

EMBEDDING_OTHER_ROOT = ROOT / "embedding_other"
EMBEDDING_OTHER_FIGS = FIGS / "embedding_other"


def test_embedding_filter_cell_type(dense_sample, image_comparer):
    mb.plot.embedding_filter(dense_sample, mask=dense_sample.obs["cell_type"] == "ct1", group="A")

    save_and_compare_images = image_comparer(EMBEDDING_OTHER_ROOT, EMBEDDING_OTHER_FIGS, tol=15)
    save_and_compare_images("filter")


def test_embedding_zoom(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    fig = mb.plot.embedding_zoom(
        dense_sample,
        mask=list(set(cell_contact.keys()).union(set(itertools.chain.from_iterable(cell_contact.values())))),
        left_pct=0.1,
        top_pct=0.2,
        height_pct=0.47,
        width_pct=0.3,
        group="cell_type",
        show=False,
    )

    fig.set_size_inches((6.4, 3.2))

    save_and_compare_images = image_comparer(EMBEDDING_OTHER_ROOT, EMBEDDING_OTHER_FIGS, tol=15)
    save_and_compare_images("zoom")
