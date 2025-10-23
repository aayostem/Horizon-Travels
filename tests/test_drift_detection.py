# tests/test_drift_detection.py
import pytest
import pandas as pd
import numpy as np
from src.monitoring.data_drift import DataDriftDetector


class TestDataDriftDetection:
    def setup_method(self):
        # Create reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.choice(["A", "B", "C"], 1000),
                "feature3": np.random.exponential(2, 1000),
            }
        )

        self.detector = DataDriftDetector(self.reference_data)

    def test_psi_calculation(self):
        # Test with identical distributions
        psi = self.detector.calculate_psi(
            self.reference_data["feature1"], self.reference_data["feature1"]
        )
        assert psi == 0.0

    def test_drift_detection_no_drift(self):
        # Test with same data (no drift)
        results = self.detector.detect_drift(self.reference_data)
        assert not results["overall_drift"]

    def test_drift_detection_with_drift(self):
        # Create drifted data
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 1000),  # Different distribution
                "feature2": np.random.choice(["A", "B", "C"], 1000, p=[0.8, 0.1, 0.1]),
                "feature3": np.random.exponential(2, 1000),
            }
        )

        results = self.detector.detect_drift(drifted_data)
        # Should detect drift in feature1 and feature2
        assert results["features"]["feature1"]["drift_detected"]
        assert results["features"]["feature2"]["drift_detected"]
