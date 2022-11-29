import numpy as np
import pandas as pd

import monkeybread as mb

dense_sample_3_radius = pd.DataFrame(
    {
        "ct2": [2, 5, 6, 7, 2, 5, 5, 5, 7, 4, 6, 7, 2, 6, 3, 1],
        "ct1": [0, 3, 3, 3, 2, 1, 4, 3, 4, 4, 5, 3, 0, 2, 1, 1],
        "cell_type": ["ct1"] * 6 + ["ct2"] * 10,
    }
)


def test_neighborhood_dense_nonnormalized(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(dense_sample, groupby="cell_type", radius=3, normalize_counts=False)
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert np.allclose(ad_neighborhood.obsm["X_spatial"], dense_sample.obsm["X_spatial"])
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"].cat.categories)
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(ad_neighborhood.X, dense_sample_3_radius[["ct2", "ct1"]].to_numpy())
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"], np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)
    )


def test_neighborhood_dense_normalized(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(dense_sample, groupby="cell_type", radius=3)
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert np.allclose(ad_neighborhood.obsm["X_spatial"], dense_sample.obsm["X_spatial"])
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"].cat.categories)
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(
        ad_neighborhood.X,
        np.apply_along_axis(
            lambda arr: arr / np.sum(arr), axis=1, arr=dense_sample_3_radius[["ct2", "ct1"]].to_numpy()
        ),
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"], np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)
    )


def test_neighborhood_dense_nonnormalized_neighborhood_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample, groupby="cell_type", radius=3, normalize_counts=False, neighborhood_groups=["ct2"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert np.allclose(ad_neighborhood.obsm["X_spatial"], dense_sample.obsm["X_spatial"])
    assert set(ad_neighborhood.var.index) == {"ct2"}
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(ad_neighborhood.X, dense_sample_3_radius[["ct2"]].to_numpy())
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"], np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)
    )


def test_neighborhood_dense_normalized_neighborhood_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample, groupby="cell_type", radius=3, neighborhood_groups=["ct2"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(dense_sample.obs["cell_type"])
    assert np.allclose(ad_neighborhood.obsm["X_spatial"], dense_sample.obsm["X_spatial"])
    assert set(ad_neighborhood.var.index) == {"ct2"}
    assert ad_neighborhood.obs.index.equals(dense_sample.obs.index)
    assert np.allclose(
        ad_neighborhood.X,
        np.apply_along_axis(
            lambda arr: arr / np.sum(arr), axis=1, arr=dense_sample_3_radius[["ct2", "ct1"]].to_numpy()
        )[:, [0]],
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"], np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)
    )


def test_neighborhood_dense_nonnormalized_subset_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(
        dense_sample, groupby="cell_type", radius=3, normalize_counts=False, subset_groups=["ct2"]
    )
    assert ad_neighborhood.obs["cell_type"].equals(
        dense_sample[dense_sample.obs["cell_type"] == "ct2"].obs["cell_type"]
    )
    assert np.allclose(
        ad_neighborhood.obsm["X_spatial"], dense_sample[dense_sample.obs["cell_type"] == "ct2"].obsm["X_spatial"]
    )
    assert set(ad_neighborhood.obs["cell_type"]) == {"ct2"}
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"])
    assert ad_neighborhood.obs.index.equals(dense_sample[dense_sample.obs["cell_type"] == "ct2"].obs.index)
    assert np.allclose(
        ad_neighborhood[:, ["ct2", "ct1"]].X,
        dense_sample_3_radius[["ct2", "ct1"]][dense_sample_3_radius["cell_type"] == "ct2"].to_numpy(),
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)[dense_sample_3_radius["cell_type"] == "ct2"],
    )


def test_neighborhood_dense_normalized_subset_groups(dense_sample):
    ad_neighborhood = mb.calc.neighborhood_profile(dense_sample, groupby="cell_type", radius=3, subset_groups=["ct2"])
    assert ad_neighborhood.obs["cell_type"].equals(
        dense_sample[dense_sample.obs["cell_type"] == "ct2"].obs["cell_type"]
    )
    assert np.allclose(
        ad_neighborhood.obsm["X_spatial"], dense_sample[dense_sample.obs["cell_type"] == "ct2"].obsm["X_spatial"]
    )
    assert set(ad_neighborhood.obs["cell_type"]) == {"ct2"}
    assert set(ad_neighborhood.var.index) == set(dense_sample.obs["cell_type"])
    assert ad_neighborhood.obs.index.equals(dense_sample[dense_sample_3_radius["cell_type"] == "ct2"].obs.index)
    assert np.allclose(
        ad_neighborhood[:, ["ct2", "ct1"]].X,
        np.apply_along_axis(
            lambda arr: arr / np.sum(arr),
            axis=1,
            arr=dense_sample_3_radius[["ct2", "ct1"]][dense_sample_3_radius["cell_type"] == "ct2"].to_numpy(),
        ),
    )
    assert ad_neighborhood.uns["neighbor_radius"] == 3
    assert np.allclose(
        ad_neighborhood.obs["n_neighbors"],
        np.sum(dense_sample_3_radius[["ct2", "ct1"]].to_numpy(), axis=1)[dense_sample_3_radius["cell_type"] == "ct2"],
    )
