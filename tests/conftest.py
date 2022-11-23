from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def create_sample(data: Dict[str, List[List[float]]], dims: Optional[Tuple[float, float]] = None):
    num_cells = sum(len(v) for v in data.values())
    sample = ad.AnnData(
        X=np.array([np.array([abs(j - i) for j in range(65, 91)]) for i in range(num_cells)]),
        obsm={
            "X_spatial": np.array([coords for key, val in data.items() for coords in val]),
        },
        obs={
            "cell_type": pd.Categorical([ct for ct, val in data.items() for _ in val]),
        },
        oidx=np.array([str(i) for i in range(num_cells)]),
        vidx=np.array([chr(i) for i in range(65, 91)]),
        dtype=np.dtype(np.float32),
    )
    if dims is not None:
        sample.obs["width"] = np.full(sample.shape[0], fill_value=dims[0])
        sample.obs["height"] = np.full(sample.shape[0], fill_value=dims[1])
    return sample


@pytest.fixture(scope="module")
def dense_sample():
    return create_sample(
        {
            "ct1": [
                [2, 2],
                [2.5, 6.5],
                [3, 6],
                [4, 5],
                [5, 8],
                [7, 5],
            ],
            "ct2": [[3, 4], [3, 5], [4, 4], [4, 7], [5, 6], [5.5, 5], [6, 2], [6, 4], [8, 5], [8, 7]],
        },
        dims=(2, 2),
    )


@pytest.fixture(scope="module")
def sparse_sample():
    return create_sample(
        {"ct1": [[1, 8], [6, 1]], "ct2": [[1, 5], [2, 7], [2, 8], [4, 3], [5, 1], [6, 2]]}, dims=(1, 3)
    )


@pytest.fixture(scope="module")
def ct3_sample():
    return create_sample(
        {"ct1": [[1, 8], [6, 1]], "ct2": [[1, 5], [4, 3]], "ct3": [[2, 7], [2, 8], [5, 1], [6, 2]]}, dims=(1, 3)
    )
