from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusevirome import (
    FIG_DIR,
    load_data,
    stage1_null_removal,
    stage2_fused_exclusion,
)


class PipelineTests(unittest.TestCase):
    def test_data_loading_and_feature_refinement(self):
        x_train, y_train, x_test, y_test = load_data()

        self.assertEqual(x_train.shape, (120, 35))
        self.assertEqual(x_test.shape, (179, 35))
        self.assertEqual(len(y_train), 120)
        self.assertEqual(len(y_test), 179)
        self.assertNotIn("Y_label_max", x_train.columns)
        self.assertNotIn("qPCR", x_test.columns)

        stage1_removed = stage1_null_removal(x_train)
        x_stage1 = x_train.drop(columns=stage1_removed)
        stage2_removed = stage2_fused_exclusion(x_stage1)
        x_final = x_stage1.drop(columns=stage2_removed)

        self.assertEqual(x_final.shape[1], 29)

    def test_generated_figures_are_separate_from_reference_figures(self):
        output_path = Path(FIG_DIR)
        self.assertEqual(output_path.parts[-2:], ("outputs", "figures"))


if __name__ == "__main__":
    unittest.main()
