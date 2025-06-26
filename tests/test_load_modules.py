import unittest


class LoadHelpTestClass(unittest.TestCase):
    def test_models(self) -> None:
        import models

        self.assertTrue(hasattr(models, "BrnnModel"), "Models not loaded")

    def test_utils(self) -> None:
        import utils

        self.assertTrue(hasattr(utils, "F64_A"), "Utils not loaded")

    def test_utils_fed(self) -> None:
        import utils.fed as fed_utils

        self.assertTrue(hasattr(fed_utils, "Cube"), "Fed utils not loaded")
