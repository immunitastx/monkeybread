import numpy as np

import monkeybread as mb


def test_shortest_distance_dense_ct1_ct2(dense_sample):
    ct1_indices, distances, ct2_indices = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2").T
    expected_ct1_indices = dense_sample[dense_sample.obs["cell_type"] == "ct1"].obs.index
    expected_distances = [np.sqrt(5), np.sqrt(np.square(0.5) + np.square(1.5)), 1, 1, np.sqrt(2), 1]
    expected_ct2_indices = ["6", "7", "7", "7", "9", "14"]
    assert np.array_equal(ct1_indices, expected_ct1_indices)
    assert np.array_equal(ct2_indices, expected_ct2_indices)
    assert np.allclose(distances.astype(float), expected_distances)


def test_shortest_distance_dense_ct2_ct1(dense_sample):
    ct2_indices, distances, ct1_indices = mb.calc.shortest_distances(dense_sample, "cell_type", "ct2", "ct1").T
    expected_ct2_indices = dense_sample[dense_sample.obs["cell_type"] == "ct2"].obs.index
    expected_distances = [np.sqrt(2), 1, 1, np.sqrt(2), np.sqrt(2), 1.5, np.sqrt(10), np.sqrt(2), 1, np.sqrt(5)]
    expected_ct1_indices = ["3", "2", "3", "2", "3", "3", "5", "5", "5", "5"]
    assert np.array_equal(ct2_indices, expected_ct2_indices)
    assert np.array_equal(ct1_indices, expected_ct1_indices)
    assert np.allclose(distances.astype(float), expected_distances)


def test_shortest_distance_dense_ct1_ct2_insignificance(dense_sample):
    shortest_distances = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2")
    expected_distances, threshold, p_val = mb.stat.shortest_distances(
        dense_sample, "cell_type", "ct1", "ct2", actual=shortest_distances, threshold=2
    )
    assert threshold == 2
    assert p_val == 1.0  # Only 2 cell types, so permutation does not change labels
    assert np.allclose(expected_distances, np.tile(shortest_distances.T[1].astype(float), 100))


def test_shortest_distance_dense_ct1_ct2_no_threshold(dense_sample):
    shortest_distances = mb.calc.shortest_distances(dense_sample, "cell_type", "ct1", "ct2")
    expected_distances = mb.stat.shortest_distances(dense_sample, "cell_type", "ct1", "ct2", actual=shortest_distances)
    assert np.allclose(expected_distances, np.tile(shortest_distances.T[1].astype(float), 100))


def test_shortest_distance_sparse_ct1_ct2(sparse_sample):
    ct1_indices, distances, ct2_indices = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct1", "ct2").T
    expected_ct1_indices = sparse_sample[sparse_sample.obs["cell_type"] == "ct1"].obs.index
    expected_distances = [1, 1]
    expected_ct2_indices = ["4", "6"]
    assert np.array_equal(ct1_indices, expected_ct1_indices)
    assert np.array_equal(ct2_indices, expected_ct2_indices)
    assert np.allclose(distances.astype(float), expected_distances)


def test_shortest_distance_sparse_ct2_ct1(sparse_sample):
    ct2_indices, distances, ct1_indices = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct2", "ct1").T
    expected_ct2_indices = sparse_sample[sparse_sample.obs["cell_type"] == "ct2"].obs.index
    expected_distances = [3, np.sqrt(2), 1, np.sqrt(8), 1, 1]
    expected_ct1_indices = ["0", "0", "0", "1", "1", "1"]
    assert np.array_equal(ct2_indices, expected_ct2_indices)
    assert np.array_equal(ct1_indices, expected_ct1_indices)
    assert np.allclose(distances.astype(float), expected_distances)


def test_shortest_distance_sparse_ct1_ct2_insignificance(sparse_sample):
    shortest_distances = mb.calc.shortest_distances(sparse_sample, "cell_type", "ct1", "ct2")
    expected_distances, threshold, p_val = mb.stat.shortest_distances(
        sparse_sample, "cell_type", "ct1", "ct2", actual=shortest_distances, threshold=5
    )
    assert threshold == 5
    assert p_val == 1.0  # Only 2 cell types, so permutation does not change labels
    assert np.allclose(expected_distances, np.tile(shortest_distances.T[1].astype(float), 100))


def test_shortest_distance_ct3_ct1(ct3_sample):
    ct3_indices, distances, ct1_indices = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct3", "ct1").T
    expected_ct3_indices = ct3_sample[ct3_sample.obs["cell_type"] == "ct3"].obs.index
    expected_distances = [np.sqrt(2), 1, 1, 1]
    expected_ct1_indices = ["0", "0", "1", "1"]
    assert np.array_equal(ct3_indices, expected_ct3_indices)
    assert np.array_equal(ct1_indices, expected_ct1_indices)
    assert np.allclose(distances.astype(float), expected_distances)


def test_shortest_distance_ct3_ct1_significance(ct3_sample):
    shortest_distances = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct3", "ct1")
    expected_distances, threshold, p_val = mb.stat.shortest_distances(
        ct3_sample, "cell_type", "ct3", "ct1", actual=shortest_distances, threshold=1
    )
    assert threshold == 1
    assert p_val < 0.20


def test_shortest_distance_ct1_ct3_insignificance(ct3_sample):
    shortest_distances = mb.calc.shortest_distances(ct3_sample, "cell_type", "ct1", "ct3")
    expected_distances, threshold, p_val = mb.stat.shortest_distances(
        ct3_sample, "cell_type", "ct1", "ct3", actual=shortest_distances, threshold=1
    )
    assert threshold == 1
    assert p_val > 0.50
