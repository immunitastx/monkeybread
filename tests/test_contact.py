import unittest
import anndata
import monkeybread as mb
from abc import ABC, abstractmethod
import sample_data


class TestCellContact(unittest.TestCase, ABC):
    @property
    @abstractmethod
    def adata(self) -> anndata.AnnData:
        pass

    def assert_total_contact(self, group1, group2, expected_num, radius = None, **kwargs):
        observed_contact = mb.calc.cell_contact(self.adata, "cell_type", group1, group2,
                                                radius = radius, **kwargs)
        observed_num = sum(len(v) for v in observed_contact.values())
        observed_coords = {
            tuple(self.adata[key].obsm["X_spatial"][0]): list(map(
                lambda cell_id: tuple(self.adata[cell_id].obsm["X_spatial"][0]),
                values
            )) for key, values in observed_contact.items()
        }
        self.assertEqual(
            observed_num,
            expected_num,
            f'Expected {expected_num} contacts, found {observed_num}.\n{observed_coords}'
        )


class DenseSampleTest(TestCellContact):

    @property
    def adata(self):
        return sample_data.dense_sample

    def test_contact_radius_0(self):
        super().assert_total_contact("DC", "T", 0, radius = 0)

    def test_contact_radius_1(self):
        super().assert_total_contact("DC", "T", 4, radius = 1)

    def test_contact_radius_2(self):
        super().assert_total_contact("DC", "T", 17, radius = 2)

    def test_contact_radius_3(self):
        super().assert_total_contact("DC", "T", 27, radius = 3)

    def test_contact_radius_reflexivity(self):
        super().assert_total_contact("T", "DC", 4, radius = 1)
        super().assert_total_contact("T", "DC", 17, radius = 2)
        super().assert_total_contact("T", "DC", 27, radius = 3)

    def test_contact_radius_self_count(self):
        super().assert_total_contact("DC", "DC", 1, radius = 1)

    def test_auto_radius_calculation(self):
        super().assert_total_contact("DC", "T", 17)


class SparseSampleTest(TestCellContact):

    @property
    def adata(self):
        return sample_data.sparse_sample

    def test_contact_radius_0(self):
        super().assert_total_contact("DC", "T", 0, radius = 0)

    def test_contact_radius_1(self):
        super().assert_total_contact("DC", "T", 3, radius = 1)

    def test_contact_radius_2(self):
        super().assert_total_contact("DC", "T", 4, radius = 2)

    def test_contact_radius_3(self):
        super().assert_total_contact("DC", "T", 6, radius = 3)

    def test_contact_radius_reflexivity(self):
        super().assert_total_contact("T", "DC", 3, radius = 1)
        super().assert_total_contact("T", "DC", 4, radius = 2)
        super().assert_total_contact("T", "DC", 6, radius = 3)

    def test_contact_radius_self_count(self):
        super().assert_total_contact("T", "T", 2, radius = 2)

    def test_auto_radius_calculation(self):
        super().assert_total_contact("DC", "T", 4)


if __name__ == '__main__':
    test_classes = [DenseSampleTest, SparseSampleTest]
    suite = map(unittest.defaultTestLoader.loadTestsFromTestCase, test_classes)
    unittest.TextTestRunner().run(unittest.TestSuite(suite))
