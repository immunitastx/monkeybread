import pytest
import monkeybread as mb


def assert_total_contact(adata, group1, group2, expected_num, radius = None, **kwargs):
    observed_contact = mb.calc.cell_contact(adata, "cell_type", group1, group2,
                                            radius = radius, **kwargs)
    observed_num = sum(len(v) for v in observed_contact.values())
    assert observed_num == expected_num


def assert_unique_contact(adata, group1, group2, expected_num, radius = None, **kwargs):
    g1 = adata[[c in group1 for c in adata.obs["cell_type"]]].obs.index
    g2 = adata[[c in group2 for c in adata.obs["cell_type"]]].obs.index
    observed_contact = mb.calc.cell_contact(adata, "cell_type", group1, group2,
                                            radius = radius, **kwargs)
    observed_num = sum(len(v) for v in observed_contact.values()) - \
        int(0.5 * sum(0 if k not in g2 else sum(v in g1 for v in values) for
            k, values in observed_contact.items()))
    assert observed_num == expected_num


def test_dense_contact_radius_0(dense_sample):
    assert_total_contact(dense_sample, "DC", "T", 0, radius = 0)


def test_dense_contact_radius_1(dense_sample):
    assert_total_contact(dense_sample, "DC", "T", 4, radius = 1)


def test_dense_contact_radius_2(dense_sample):
    assert_total_contact(dense_sample, "DC", "T", 17, radius = 2)


def test_dense_contact_radius_3(dense_sample):
    assert_total_contact(dense_sample, "DC", "T", 27, radius = 3)


def test_dense_contact_radius_reflexivity(dense_sample):
    assert_total_contact(dense_sample, "T", "DC", 4, radius = 1)
    assert_total_contact(dense_sample, "T", "DC", 17, radius = 2)
    assert_total_contact(dense_sample, "T", "DC", 27, radius = 3)


def test_dense_contact_radius_self_count(dense_sample):
    assert_total_contact(dense_sample, "DC", "DC", 2, radius = 1)
    assert_unique_contact(dense_sample, "DC", "DC", 1, radius = 1)


def test_dense_auto_radius_calculation(dense_sample):
    assert_total_contact(dense_sample, "DC", "T", 17)


def test_dense_group_contact(dense_sample):
    assert_total_contact(dense_sample, ["DC", "T"], "T", 8, radius = 1)
    assert_unique_contact(dense_sample, ["DC", "T"], "T", 6, radius = 1)


def test_dense_insignificance(dense_sample):
    actual_contact = mb.calc.cell_contact(dense_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(dense_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 1)
    assert p_val > 0.05


def test_dense_significance(dense_sample):
    actual_contact = mb.calc.cell_contact(dense_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(dense_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 50)
    assert p_val < 0.05


def test_sparse_contact_radius_0(sparse_sample):
    assert_total_contact(sparse_sample, "DC", "T", 0, radius = 0)


def test_sparse_contact_radius_1(sparse_sample):
    assert_total_contact(sparse_sample, "DC", "T", 3, radius = 1)


def test_sparse_contact_radius_2(sparse_sample):
    assert_total_contact(sparse_sample, "DC", "T", 4, radius = 2)


def test_sparse_contact_radius_3(sparse_sample):
    assert_total_contact(sparse_sample, "DC", "T", 6, radius = 3)


def test_sparse_contact_radius_reflexivity(sparse_sample):
    assert_total_contact(sparse_sample, "T", "DC", 3, radius = 1)
    assert_total_contact(sparse_sample, "T", "DC", 4, radius = 2)
    assert_total_contact(sparse_sample, "T", "DC", 6, radius = 3)


def test_sparse_contact_radius_self_count(sparse_sample):
    assert_total_contact(sparse_sample, "T", "T", 4, radius = 2)
    assert_unique_contact(sparse_sample, "T", "T", 2, radius = 2)


def test_sparse_auto_radius_calculation(sparse_sample):
    assert_total_contact(sparse_sample, "DC", "T", 4)


def test_sparse_significance(sparse_sample):
    actual_contact = mb.calc.cell_contact(sparse_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(sparse_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 10)
    assert p_val < 0.05
