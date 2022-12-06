import numpy as np

import monkeybread as mb


def test_ligand_receptor_score(dense_sample):
    contacts = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    lr_scores = mb.calc.ligand_receptor_score(dense_sample, contacts, [("A", "B"), ("C", "D")])

    expected_ab_score = (np.sqrt(0.64 * 3.1) + np.sqrt(7.39 * 3.1) + np.sqrt(7.39 * 2.5) + np.sqrt(6.4 * 4.66)) / 4
    expected_cd_score = (np.sqrt(4.06 * 5.21) + np.sqrt(7.19 * 5.21) + np.sqrt(7.19 * 6.47) + np.sqrt(0.18 * 6.46)) / 4

    assert np.allclose(lr_scores[("A", "B")], expected_ab_score, atol=0.01)
    assert np.allclose(lr_scores[("C", "D")], expected_cd_score, atol=0.01)
