import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_preparation import parse_percent, parse_size


class ParsingTests(unittest.TestCase):
    def test_parse_size(self):
        cases = [
            ("1.2 Gbp", 1.2e9),
            ("3.5 Mbp", 3.5e6),
            ("500 Kbp", 5.0e5),
            ("42 bp", 42.0),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertAlmostEqual(parse_size(value), expected)

    def test_parse_size_handles_missing_value(self):
        self.assertTrue(math.isnan(parse_size(None)))

    def test_parse_percent(self):
        cases = [("95.5%", 95.5), (12, 12.0), (0.25, 0.25)]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertAlmostEqual(parse_percent(value), expected)


if __name__ == "__main__":
    unittest.main()
