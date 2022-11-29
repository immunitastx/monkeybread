import monkeybread as mb


def assert_total_contact(adata, group1, group2, expected_num, radius=None, **kwargs):
    observed_contact = mb.calc.cell_contact(adata, "cell_type", group1, group2, radius=radius, **kwargs)
    observed_num = sum(len(v) for v in observed_contact.values())
    assert observed_num == expected_num


def assert_unique_contact(adata, group1, group2, expected_num, radius=None, **kwargs):
    g1 = adata[[c in group1 for c in adata.obs["cell_type"]]].obs.index
    g2 = adata[[c in group2 for c in adata.obs["cell_type"]]].obs.index
    observed_contact = mb.calc.cell_contact(adata, "cell_type", group1, group2, radius=radius, **kwargs)
    observed_num = mb.util.contact_count(observed_contact, g1, g2)
    assert observed_num == expected_num


def test_contact_count():
    d1 = {1: {2}, 2: {1, 3}, 3: {2}}
    assert mb.util.contact_count(d1, [1, 2], [2, 3]) == 2
    d2 = {1: {2}, 2: {1, 3}, 3: {2}}
    assert mb.util.contact_count(d2, [1], [3]) == 0
    d3 = {1: {2, 3, 4}, 2: {1, 4, 5}, 3: {1, 4, 6}, 4: {1, 2, 3, 5, 6}, 5: {2, 4, 6}, 6: {3, 4, 5, 7, 8}, 7: {6, 8}}
    assert mb.util.contact_count(d3, [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 8]) == 13


def test_dense_contact_radius_0(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct2", 0, radius=0)


def test_dense_contact_radius_1(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct2", 4, radius=1)


def test_dense_contact_radius_2(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct2", 17, radius=2)


def test_dense_contact_radius_3(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct2", 27, radius=3)


def test_dense_contact_radius_reflexivity(dense_sample):
    assert_total_contact(dense_sample, "ct2", "ct1", 4, radius=1)
    assert_total_contact(dense_sample, "ct2", "ct1", 17, radius=2)
    assert_total_contact(dense_sample, "ct2", "ct1", 27, radius=3)


def test_dense_contact_radius_self_count(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct1", 2, radius=1)
    assert_unique_contact(dense_sample, "ct1", "ct1", 1, radius=1)


def test_dense_auto_radius_calculation(dense_sample):
    assert_total_contact(dense_sample, "ct1", "ct2", 17)


def test_dense_group_contact(dense_sample):
    assert_total_contact(dense_sample, ["ct1", "ct2"], "ct2", 8, radius=1)
    assert_unique_contact(dense_sample, ["ct1", "ct2"], "ct2", 6, radius=1)


def test_dense_insignificance(dense_sample):
    actual_contact = mb.calc.cell_contact(dense_sample, "cell_type", "ct1", "ct2", radius=2)
    perm_dist, p_val = mb.stat.cell_contact(
        dense_sample, "cell_type", "ct1", "ct2", actual_contact, contact_radius=2, perm_radius=1
    )
    assert p_val > 0.05


def test_dense_significance(dense_sample):
    actual_contact = mb.calc.cell_contact(dense_sample, "cell_type", "ct1", "ct2", radius=2)
    perm_dist, p_val = mb.stat.cell_contact(
        dense_sample, "cell_type", "ct1", "ct2", actual_contact, contact_radius=2, perm_radius=50
    )
    assert p_val < 0.05


def test_sparse_contact_radius_0(sparse_sample):
    assert_total_contact(sparse_sample, "ct1", "ct2", 0, radius=0)


def test_sparse_contact_radius_1(sparse_sample):
    assert_total_contact(sparse_sample, "ct1", "ct2", 3, radius=1)


def test_sparse_contact_radius_2(sparse_sample):
    assert_total_contact(sparse_sample, "ct1", "ct2", 4, radius=2)


def test_sparse_contact_radius_3(sparse_sample):
    assert_total_contact(sparse_sample, "ct1", "ct2", 6, radius=3)


def test_sparse_contact_radius_reflexivity(sparse_sample):
    assert_total_contact(sparse_sample, "ct2", "ct1", 3, radius=1)
    assert_total_contact(sparse_sample, "ct2", "ct1", 4, radius=2)
    assert_total_contact(sparse_sample, "ct2", "ct1", 6, radius=3)


def test_sparse_contact_radius_self_count(sparse_sample):
    assert_total_contact(sparse_sample, "ct2", "ct2", 4, radius=2)
    assert_unique_contact(sparse_sample, "ct2", "ct2", 2, radius=2)


def test_sparse_auto_radius_calculation(sparse_sample):
    assert_total_contact(sparse_sample, "ct1", "ct2", 4)


def test_sparse_significance(sparse_sample):
    actual_contact = mb.calc.cell_contact(sparse_sample, "cell_type", "ct1", "ct2", radius=2)
    perm_dist, p_val = mb.stat.cell_contact(
        sparse_sample, "cell_type", "ct1", "ct2", actual_contact, contact_radius=2, perm_radius=10
    )
    assert p_val < 0.05
