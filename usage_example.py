# example_usage.py
from src.monitoring.pipeline import DriftMonitoringPipeline

# Initialize pipeline
pipeline = DriftMonitoringPipeline("configs/drift_config.yaml")

# Run monitoring
results = pipeline.run_monitoring_pipeline()

# Check results
if results["data_drift_results"]["overall_drift"]:
    print("Data drift detected! Investigate features:")
    for feature, metrics in results["data_drift_results"]["features"].items():
        if metrics["drift_detected"]:
            print(f"  - {feature}: PSI={metrics.get('psi', 'N/A')}")

for alert in results["concept_drift_alerts"]:
    print(f"Alert: {alert.severity} - {alert.message}")
