import random

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

SHORTEST_DISTANCES_ROOT = ROOT / "shortest_distances"
SHORTEST_DISTANCES_FIGS = FIGS / "shortest_distances"


def test_shortest_distances(dense_sample, image_comparer):
    distance_df = mb.calc.shortest_distances(dense_sample, groupby="cell_type", group1="ct1", group2="ct2")
    mb.plot.shortest_distances(distance_df, show=False)

    save_and_compare_images = image_comparer(SHORTEST_DISTANCES_ROOT, SHORTEST_DISTANCES_FIGS, tol=15)
    save_and_compare_images("plain_histplot")


def test_shortest_distances_expected_distribution(dense_sample, image_comparer):
    distance_df = mb.calc.shortest_distances(dense_sample, groupby="cell_type", group1="ct1", group2="ct2")
    random.seed(0)
    expected_distances = mb.stat.shortest_distances(
        dense_sample, groupby="cell_type", group1="ct1", group2="ct2", actual=distance_df, threshold=1.5
    )
    mb.plot.shortest_distances(distance_df, expected_distances=expected_distances, show=False)

    save_and_compare_images = image_comparer(SHORTEST_DISTANCES_ROOT, SHORTEST_DISTANCES_FIGS, tol=15)
    save_and_compare_images("significance_histplot")
