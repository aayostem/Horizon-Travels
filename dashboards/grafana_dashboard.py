# dashboards/grafana_dashboard.py
import json


def create_drift_dashboard():
    """Generate Grafana dashboard JSON for drift monitoring"""

    dashboard = {
        "dashboard": {
            "title": "ML Model Drift Monitoring",
            "panels": [
                {
                    "title": "Data Drift Overview",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "sum(drift_detected_features)",
                            "legendFormat": "Features with Drift",
                        }
                    ],
                },
                {
                    "title": "PSI Values by Feature",
                    "type": "bargauge",
                    "targets": [
                        {
                            "expr": 'feature_psi{feature=~"$feature"}',
                            "legendFormat": "{{feature}}",
                        }
                    ],
                },
                {
                    "title": "Model Accuracy Over Time",
                    "type": "timeseries",
                    "targets": [{"expr": "model_accuracy", "legendFormat": "Accuracy"}],
                },
            ],
        }
    }

    return json.dumps(dashboard, indent=2)
