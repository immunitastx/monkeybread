import monkeybread as mb


def test_spatial_subset(sparse_sample):
    assert len(mb.util.subset_cells(sparse_sample, "spatial", ("x", "lte", 2))) == 4


def test_spatial_subset_conditions_list(sparse_sample):
    assert (
        len(
            mb.util.subset_cells(
                sparse_sample, "spatial", [("x", "gte", 2), ("x", "lte", 6), ("y", "gte", 2), ("y", "lte", 6)]
            )
        )
        == 2
    )
