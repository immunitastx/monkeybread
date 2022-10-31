import unittest
import monkeybread as mb
import sample_data


class SparseSampleTest(unittest.TestCase):

    def test_spatial_subset(self):
        self.assertEquals(
            len(mb.util.subset_cells(sample_data.sparse_sample, "spatial", ("x", "lte", 2))),
            4
        )

    def test_spatial_subset_conditions_list(self):
        self.assertEquals(
            len(mb.util.subset_cells(sample_data.sparse_sample, "spatial", [
                ("x", "gte", 2),
                ("x", "lte", 6),
                ("y", "gte", 2),
                ("y", "lte", 6)
            ])),
            2
        )


if __name__ == '__main__':
    test_classes = [SparseSampleTest]
    suite = map(unittest.defaultTestLoader.loadTestsFromTestCase, test_classes)
    unittest.TextTestRunner().run(unittest.TestSuite(suite))
