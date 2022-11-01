import pytest
import monkeybread as mb
import sample_data


def test_spatial_subset(self):
    assert len(mb.util.subset_cells(sample_data.sparse_sample, "spatial", ("x", "lte", 2))) == 4


def test_spatial_subset_conditions_list(self):
    assert len(mb.util.subset_cells(sample_data.sparse_sample, "spatial", [
                ("x", "gte", 2),
                ("x", "lte", 6),
                ("y", "gte", 2),
                ("y", "lte", 6)
            ])) == 2
