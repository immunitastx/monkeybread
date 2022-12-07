import numpy as np

import monkeybread as mb


def test_ligand_receptor_score(dense_sample):
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(dense_sample, contacts, [("A", "B"), ("C", "D")])

    expected_ab_score = (np.sqrt(0.64 * 3.1) + np.sqrt(7.39 * 3.1) + np.sqrt(7.39 * 2.5) + np.sqrt(6.4 * 4.66)) / 4
    expected_cd_score = (np.sqrt(4.06 * 5.21) + np.sqrt(7.19 * 5.21) + np.sqrt(7.19 * 6.47) + np.sqrt(0.18 * 6.46)) / 4

    assert np.allclose(lr_scores[("A", "B")], expected_ab_score, atol=0.01)
    assert np.allclose(lr_scores[("C", "D")], expected_cd_score, atol=0.01)


def test_ligand_receptor_score_insignificance(dense_sample):
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(dense_sample, contacts, [("A", "B"), ("C", "D")])
    lr_stat_scores = mb.stat.ligand_receptor_score(dense_sample, contacts, lr_scores, n_perms=1000)

    assert lr_stat_scores[("A", "B")][1] > 0.50
    assert lr_stat_scores[("C", "D")][1] > 0.50


def test_ligand_receptor_score_significance(dense_sample):
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(dense_sample, contacts, [("G", "I"), ("K", "I"), ("L", "I")])
    lr_stat_scores = mb.stat.ligand_receptor_score(dense_sample, contacts, lr_scores, n_perms=1000)

    assert lr_stat_scores[("G", "I")][1] < 0.2
    assert lr_stat_scores[("K", "I")][1] < 0.2
    assert lr_stat_scores[("L", "I")][1] < 0.2
