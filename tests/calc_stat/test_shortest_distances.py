import numpy as np

import monkeybread as mb


def test_shortest_distance_dense_ct1_ct2(dense_sample):
    distance_df = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2")
    expected_ct1_indices = dense_sample[dense_sample.obs["cell_type"] == "ct1"].obs.index
    expected_distances = [np.sqrt(5), np.sqrt(np.square(0.5) + np.square(1.5)), 1, 1, np.sqrt(2), 1]
    expected_ct2_indices = ["6", "7", "7", "7", "9", "14"]
    assert np.array_equal(distance_df.index, expected_ct1_indices)
    assert np.array_equal(distance_df["nearest_cell"], expected_ct2_indices)
    assert np.allclose(distance_df["distance"], expected_distances)


def test_shortest_distance_dense_ct2_ct1(dense_sample):
    distance_df = mb.calc.shortest_distances(dense_sample, "cell_type", "ct2", "ct1")
    expected_ct2_indices = dense_sample[dense_sample.obs["cell_type"] == "ct2"].obs.index
    expected_distances = [np.sqrt(2), 1, 1, np.sqrt(2), np.sqrt(2), 1.5, np.sqrt(10), np.sqrt(2), 1, np.sqrt(5)]
    expected_ct1_indices = ["3", "2", "3", "2", "3", "3", "5", "5", "5", "5"]
    assert np.array_equal(distance_df.index, expected_ct2_indices)
    assert np.array_equal(distance_df["nearest_cell"], expected_ct1_indices)
    assert np.allclose(distance_df["distance"], expected_distances)


def test_shortest_distance_dense_ct1_ct2_insignificance(dense_sample):
    shortest_distances = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2")
    expected_distances, p_val = mb.stat.shortest_distances(
        dense_sample, "cell_type", "ct1", "ct2", observed=shortest_distances, threshold=2
    )
    assert p_val == 1.0  # Only 2 cell types, so permutation does not change labels
    assert np.allclose(expected_distances, np.tile(shortest_distances["distance"], 100))


def test_shortest_distance_dense_ct1_ct2_no_threshold(dense_sample):
    shortest_distances = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2")
    expected_distances = mb.stat.shortest_distances(dense_sample, "cell_type", "ct1", "ct2", observed=shortest_distances)
    assert np.allclose(expected_distances, np.tile(shortest_distances["distance"], 100))


def test_shortest_distance_sparse_ct1_ct2(sparse_sample):
    distance_df = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct1", "ct2")
    expected_ct1_indices = sparse_sample[sparse_sample.obs["cell_type"] == "ct1"].obs.index
    expected_distances = [1, 1]
    expected_ct2_indices = ["4", "6"]
    assert np.array_equal(distance_df.index, expected_ct1_indices)
    assert np.array_equal(distance_df["nearest_cell"], expected_ct2_indices)
    assert np.allclose(distance_df["distance"], expected_distances)


def test_shortest_distance_sparse_ct2_ct1(sparse_sample):
    distance_df = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct2", "ct1")
    expected_ct2_indices = sparse_sample[sparse_sample.obs["cell_type"] == "ct2"].obs.index
    expected_distances = [3, np.sqrt(2), 1, np.sqrt(8), 1, 1]
    expected_ct1_indices = ["0", "0", "0", "1", "1", "1"]
    assert np.array_equal(distance_df.index, expected_ct2_indices)
    assert np.array_equal(distance_df["nearest_cell"], expected_ct1_indices)
    assert np.allclose(distance_df["distance"], expected_distances)


def test_shortest_distance_sparse_ct1_ct2_insignificance(sparse_sample):
    shortest_distances = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct1", "ct2")
    expected_distances, p_val = mb.stat.shortest_distances(
        sparse_sample, "cell_type", "ct1", "ct2", observed=shortest_distances, threshold=5
    )
    assert p_val == 1.0  # Only 2 cell types, so permutation does not change labels
    assert np.allclose(expected_distances, np.tile(shortest_distances["distance"], 100))


def test_shortest_distance_ct3_ct1(ct3_sample):
    distance_df = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct3", "ct1")
    expected_ct3_indices = ct3_sample[ct3_sample.obs["cell_type"] == "ct3"].obs.index
    expected_distances = [np.sqrt(2), 1, 1, 1]
    expected_ct1_indices = ["0", "0", "1", "1"]
    assert np.array_equal(distance_df.index, expected_ct3_indices)
    assert np.array_equal(distance_df["nearest_cell"], expected_ct1_indices)
    assert np.allclose(distance_df["distance"], expected_distances)


def test_shortest_distance_ct3_ct1_significance(ct3_sample):
    shortest_distances = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct3", "ct1")
    expected_distances, p_val = mb.stat.shortest_distances(
        ct3_sample, "cell_type", "ct3", "ct1", observed=shortest_distances, threshold=1, n_perms=1000
    )
    assert p_val < 0.20


def test_shortest_distance_ct1_ct3_insignificance(ct3_sample):
    shortest_distances = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct1", "ct3")
    expected_distances, p_val = mb.stat.shortest_distances(
        ct3_sample, "cell_type", "ct1", "ct3", observed=shortest_distances, threshold=1, n_perms=1000
    )
    assert p_val > 0.50
