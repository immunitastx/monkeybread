import pandas as pd
import pytest
import monkeybread as mb
import numpy as np

dense_sample_3_radius = pd.DataFrame(
    {
        "T": [2, 5, 6, 7, 2, 5, 5, 5, 7, 4, 6, 7, 6, 2, 3, 1],
        "DC": [0, 3, 3, 3, 2, 1, 4, 3, 4, 4, 5, 3, 2, 0, 1, 1],
        "cell_type": ["DC"] * 6 + ["T"] * 10
    }
)


def test_neighborhood_dense_nonnormalized(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3,
        normalize_counts = False
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"].cat.categories)
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(ad_neighborhood.X, dense_sample_3_radius[["T", "DC"]].to_numpy())
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
    )


def test_neighborhood_dense_normalized(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"].cat.categories)
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(
        ad_neighborhood.X,
        np.apply_along_axis(lambda arr: arr / np.sum(arr), axis = 1,
                            arr = dense_sample_3_radius[["T", "DC"]].to_numpy())
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
    )


def test_neighborhood_dense_nonnormalized_neighborhood_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3,
        normalize_counts = False,
        neighborhood_groups = ["T"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert set(ad_neighborhood.var.index) == {"T"}
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(ad_neighborhood.X, dense_sample_3_radius[["T"]].to_numpy())
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
    )


def test_neighborhood_dense_normalized_neighborhood_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3,
        neighborhood_groups = ["T"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert set(ad_neighborhood.var.index) == {"T"}
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(
        ad_neighborhood.X,
        np.apply_along_axis(lambda arr: arr / np.sum(arr), axis = 1,
                            arr = dense_sample_3_radius[["T", "DC"]].to_numpy())[:, [0]]
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
    )


def test_neighborhood_dense_nonnormalized_subset_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3,
        normalize_counts = False,
        subset_groups = ["T"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(
        dense_sample[dense_sample.obs["cell_type"] == "T"].obs["cell_type"])
    assert set(ad_neighborhood.obs["cell_type"]) == {"T"}
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"])
    assert ad_neighborhood.obs.index.equals(
        dense_sample[dense_sample.obs["cell_type"] == "T"].obs.index)
    assert np.allclose(ad_neighborhood.X, dense_sample_3_radius[["T", "DC"]]
                       [dense_sample_3_radius["cell_type"] == "T"].to_numpy())
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
               [dense_sample_3_radius["cell_type"] == "T"]
    )


def test_neighborhood_dense_normalized_subset_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample,
        groupby = "cell_type",
        radius = 3,
        subset_groups = ["T"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(
        dense_sample[dense_sample.obs["cell_type"] == "T"].obs["cell_type"])
    assert set(ad_neighborhood.obs["cell_type"]) == {"T"}
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"])
    assert ad_neighborhood.obs.index.equals(
        dense_sample[dense_sample_3_radius["cell_type"] == "T"].obs.index)
    assert np.allclose(
        ad_neighborhood.X,
        np.apply_along_axis(lambda arr: arr / np.sum(arr), axis = 1,
                            arr = dense_sample_3_radius[["T", "DC"]]
                                [dense_sample_3_radius["cell_type"] == "T"].to_numpy())
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["T", "DC"]].to_numpy(), axis = 1)
               [dense_sample_3_radius["cell_type"] == "T"]
    )
