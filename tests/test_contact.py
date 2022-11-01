import pytest
import monkeybread as mb
import sample_data


def assert_total_contact(adata, group1, group2, expected_num, radius = None, **kwargs):
    observed_contact = mb.calc.cell_contact(adata, "cell_type", group1, group2,
                                            radius = radius, **kwargs)
    observed_num = sum(len(v) for v in observed_contact.values())
    assert observed_num == expected_num


def test_dense_contact_radius_0():
    assert_total_contact(sample_data.dense_sample, "DC", "T", 0, radius = 0)


def test_dense_contact_radius_1():
    assert_total_contact(sample_data.dense_sample, "DC", "T", 4, radius = 1)


def test_dense_contact_radius_2():
    assert_total_contact(sample_data.dense_sample, "DC", "T", 17, radius = 2)


def test_dense_contact_radius_3():
    assert_total_contact(sample_data.dense_sample, "DC", "T", 27, radius = 3)


def test_dense_contact_radius_reflexivity():
    assert_total_contact(sample_data.dense_sample, "T", "DC", 4, radius = 1)
    assert_total_contact(sample_data.dense_sample, "T", "DC", 17, radius = 2)
    assert_total_contact(sample_data.dense_sample, "T", "DC", 27, radius = 3)


def test_dense_contact_radius_self_count():
    assert_total_contact(sample_data.dense_sample, "DC", "DC", 1, radius = 1)


def test_dense_auto_radius_calculation():
    assert_total_contact(sample_data.dense_sample, "DC", "T", 17)


def test_dense_group_contact():
    assert_total_contact(["DC", "T"], "T", 6, radius = 1)


def test_dense_insignificance():
    actual_contact = mb.calc.cell_contact(sample_data.dense_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(sample_data.dense_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 1)
    assert p_val > 0.05


def test_dense_significance():
    actual_contact = mb.calc.cell_contact(sample_data.dense_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(sample_data.dense_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 50)
    assert p_val < 0.05


def test_sparse_contact_radius_0():
    assert_total_contact(sample_data.sparse_sample, "DC", "T", 0, radius = 0)


def test_sparse_contact_radius_1():
    assert_total_contact(sample_data.sparse_sample, "DC", "T", 3, radius = 1)


def test_sparse_contact_radius_2():
    assert_total_contact(sample_data.sparse_sample, "DC", "T", 4, radius = 2)


def test_sparse_contact_radius_3():
    assert_total_contact(sample_data.sparse_sample, "DC", "T", 6, radius = 3)


def test_sparse_contact_radius_reflexivity():
    assert_total_contact(sample_data.sparse_sample, "T", "DC", 3, radius = 1)
    assert_total_contact(sample_data.sparse_sample, "T", "DC", 4, radius = 2)
    assert_total_contact(sample_data.sparse_sample, "T", "DC", 6, radius = 3)


def test_sparse_contact_radius_self_count():
    assert_total_contact(sample_data.sparse_sample, "T", "T", 2, radius = 2)


def test_sparse_auto_radius_calculation():
    assert_total_contact(sample_data.sparse_sample, "DC", "T", 4)


def test_sparse_significance():
    actual_contact = mb.calc.cell_contact(sample_data.sparse_sample, "cell_type", "DC", "T",
                                          radius = 2)
    perm_dist, p_val = mb.stat.cell_contact(sample_data.sparse_sample, "cell_type", "DC", "T",
                                            actual_contact, contact_radius = 2, perm_radius = 5)
    assert p_val < 0.05
