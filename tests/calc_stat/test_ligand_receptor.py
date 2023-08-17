import numpy as np

import monkeybread as mb


def test_ligand_receptor_score(sample_with_expression):
    contacts = mb.calc.cell_neighbors(sample_with_expression, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(sample_with_expression, contacts, [("A", "B"), ("A", "C")])

    expected_ab_score = np.sum(np.sqrt([i * i for i in range(10)])) / 10
    expected_ac_score = np.sum(np.sqrt([i * (10 - i) for i in range(10)])) / 10

    assert np.allclose(lr_scores[("A", "B")], expected_ab_score, atol=0.01)
    assert np.allclose(lr_scores[("A", "C")], expected_ac_score, atol=0.01)


def test_ligand_receptor_score_insignificance(sample_with_expression):
    contacts = mb.calc.cell_neighbors(sample_with_expression, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(sample_with_expression, contacts, ("A", "C"))
    lr_stat_scores = mb.stat.ligand_receptor_score(sample_with_expression, contacts, lr_scores, n_perms=1000)

    assert lr_stat_scores[("A", "C")][1] > 0.50


def test_ligand_receptor_score_significance(sample_with_expression):
    contacts = mb.calc.cell_neighbors(sample_with_expression, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(sample_with_expression, contacts, ("A", "B"))
    lr_stat_scores = mb.stat.ligand_receptor_score(sample_with_expression, contacts, lr_scores, n_perms=1000)

    assert lr_stat_scores[("A", "B")][1] < 0.05
