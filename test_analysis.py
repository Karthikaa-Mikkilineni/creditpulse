import unittest
import numpy as np
import pandas as pd
from train import REQUIRED, features, review_capacity


class AnalysisTests(unittest.TestCase):
    def test_review_selects_by_prediction(self):
        result = review_capacity([1, 0, 1, 0], [.8, .2, .9, .1], .5)
        self.assertEqual(result["review_count"], 2)
        self.assertEqual(result["default_capture"], 1.)

    def test_zero_default_denominator(self):
        self.assertIsNone(review_capacity([0, 0], [.2, .3], .5)["default_capture"])

    def test_bad_fraction(self):
        with self.assertRaises(ValueError):
            review_capacity([1], [.8], 0)

    def test_mismatched_predictions(self):
        with self.assertRaises(ValueError):
            review_capacity([1], [], .1)

    def test_missing_schema(self):
        with self.assertRaisesRegex(ValueError, "Missing required fields"):
            features(pd.DataFrame({"loan_status": [0, 1]}))

    def test_label_excluded_and_zero_denominators_bounded(self):
        frame = pd.DataFrame({c: [0., 1.] for c in REQUIRED})
        x, y = features(frame)
        self.assertNotIn("loan_status", x)
        self.assertTrue(np.isfinite(x.debt_burden).all())
        self.assertEqual(y.tolist(), [0., 1.])


if __name__ == "__main__":
    unittest.main()
