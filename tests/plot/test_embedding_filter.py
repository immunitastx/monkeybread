import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

EMBEDDING_FILTER_ROOT = ROOT / "embedding_filter"
EMBEDDING_FILTER_FIGS = FIGS / "embedding_filter"


def test_embedding_filter_cell_type(dense_sample, image_comparer):
    mb.plot.embedding_filter(dense_sample, mask=dense_sample.obs["cell_type"] == "ct1", group="A")

    save_and_compare_images = image_comparer(EMBEDDING_FILTER_ROOT, EMBEDDING_FILTER_FIGS, tol=15)
    save_and_compare_images("cell_type")
