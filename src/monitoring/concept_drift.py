# src/monitoring/concept_drift.py
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DriftAlert:
    timestamp: str
    drift_type: str
    severity: str
    feature: str = None
    metric_value: float = None
    threshold: float = None
    message: str = ""


class DDMDetector:
    """Drift Detection Method (DDM) for concept drift"""

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self):
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.p_cursor = 1.0
        self.s_cursor = 0.0
        self.n = 0
        self.errors = 0

    def update(self, prediction: float, actual: float, threshold: float = 0.5):
        """Update detector with new prediction"""
        self.n += 1
        error = 1 if abs(prediction - actual) > threshold else 0
        self.errors += error

        p = self.errors / self.n
        s = np.sqrt(p * (1 - p) / self.n)

        # Update minimum values
        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s

        # Check for drift
        if p + s >= self.p_min + self.drift_level * self.s_min:
            return "drift"
        elif p + s >= self.p_min + self.warning_level * self.s_min:
            return "warning"

        return "stable"


class ADWINDetector:
    """ADaptive WINdowing for concept drift detection"""

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.max_window_size = 1000

    def update(self, value: float) -> str:
        """Update with new value and check for drift"""
        self.window.append(value)

        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        if len(self.window) < 2:
            return "stable"

        # Check for drift by comparing sub-windows
        for split_point in range(1, len(self.window)):
            left = self.window[:split_point]
            right = self.window[split_point:]

            mean_left = np.mean(left)
            mean_right = np.mean(right)

            # Calculate drift threshold
            n_total = len(self.window)
            n_left = len(left)
            n_right = len(right)

            drift_threshold = np.sqrt(
                (1 / (2 * n_left) + 1 / (2 * n_right))
                * np.log(4 * n_total / self.delta)
            )

            if abs(mean_left - mean_right) > drift_threshold:
                # Drift detected, reset window
                self.window = self.window[split_point:]
                return "drift"

        return "stable"


class ConceptDriftMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.ddm_detector = DDMDetector()
        self.adwin_detector = ADWINDetector()
        self.alerts: List[DriftAlert] = []

    def monitor_performance(
        self, y_true: List[float], y_pred: List[float], timestamp: str
    ) -> List[DriftAlert]:
        """Monitor model performance for concept drift"""
        alerts = []

        # Calculate performance metrics
        accuracy = np.mean(np.array(y_true) == (np.array(y_pred) > 0.5))

        # DDM detection
        for true, pred in zip(y_true, y_pred):
            ddm_result = self.ddm_detector.update(pred, true)
            if ddm_result in ["warning", "drift"]:
                alert = DriftAlert(
                    timestamp=timestamp,
                    drift_type="concept_drift",
                    severity=ddm_result,
                    metric_value=accuracy,
                    message=f"DDM detected {ddm_result} in model performance",
                )
                alerts.append(alert)

        # ADWIN detection on accuracy
        adwin_result = self.adwin_detector.update(accuracy)
        if adwin_result == "drift":
            alert = DriftAlert(
                timestamp=timestamp,
                drift_type="concept_drift",
                severity="high",
                metric_value=accuracy,
                message="ADWIN detected significant concept drift",
            )
            alerts.append(alert)

        self.alerts.extend(alerts)
        return alerts
