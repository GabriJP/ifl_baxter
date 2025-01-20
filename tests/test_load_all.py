import unittest

from click.testing import CliRunner

from cen import cli as cen_cli
from fed import cli as fed_cli


class LoadHelpTestClass(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner(env=dict(WANDB_MODE="disabled"))

    def test_cen(self) -> None:
        res = self.runner.invoke(cen_cli, ["--help"])
        self.assertIsNone(res.exception)
        self.assertEqual(res.exit_code, 0)

    def test_fed(self) -> None:
        res = self.runner.invoke(fed_cli, ["--help"])
        self.assertIsNone(res.exception)
        self.assertEqual(res.exit_code, 0)
