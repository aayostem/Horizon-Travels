# src/monitoring/data_drift.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_results = {}
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                     bucket_type: str = 'bins', buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Clip to avoid infinite values
        eps = 1e-4
        expected = expected.clip(lower=eps)
        actual = actual.clip(lower=eps)
        
        if bucket_type == 'bins':
            breakpoints = np.arange(0, 1 + 1/buckets, 1/buckets)
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        else:
            # Use quantiles
            breakpoints = np.percentile(expected, np.arange(0, 100 + 100/buckets, 100/buckets))
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Calculate PSI
        psi = np.sum((expected_percents - actual_percents) * 
                    np.log(expected_percents / actual_percents))
        return psi
    
    def ks_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for distribution comparison"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def chi_square_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Chi-square test for categorical variables"""
        # Create contingency table
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Align indices
        all_categories = ref_counts.index.union(curr_counts.index)
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
        
        # Perform chi-square test
        statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
        return statistic, p_value
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect data drift for all features"""
        drift_results = {
            'features': {},
            'overall_drift': False,
            'summary': {
                'features_with_drift': 0,
                'total_features': len(self.reference_data.columns)
            }
        }
        
        for column in self.reference_data.columns:
            ref_series = self.reference_data[column]
            curr_series = current_data[column]
            
            # Handle missing values
            ref_series = ref_series.dropna()
            curr_series = curr_series.dropna()
            
            if len(ref_series) == 0 or len(curr_series) == 0:
                continue
            
            feature_results = {}
            
            # Numerical features
            if pd.api.types.is_numeric_dtype(ref_series):
                # PSI
                psi_value = self.calculate_psi(ref_series, curr_series)
                feature_results['psi'] = psi_value
                feature_results['psi_drift'] = psi_value > self.threshold
                
                # KS Test
                ks_stat, ks_pvalue = self.ks_test(ref_series, curr_series)
                feature_results['ks_statistic'] = ks_stat
                feature_results['ks_pvalue'] = ks_pvalue
                feature_results['ks_drift'] = ks_pvalue < self.threshold
                
            # Categorical features
            else:
                # Chi-square test
                chi_stat, chi_pvalue = self.chi_square_test(ref_series, curr_series)
                feature_results['chi_statistic'] = chi_stat
                feature_results['chi_pvalue'] = chi_pvalue
                feature_results['chi_drift'] = chi_pvalue < self.threshold
            
            # Overall drift for this feature
            feature_results['drift_detected'] = any([
                feature_results.get('psi_drift', False),
                feature_results.get('ks_drift', False),
                feature_results.get('chi_drift', False)
            ])
            
            if feature_results['drift_detected']:
                drift_results['summary']['features_with_drift'] += 1
            
            drift_results['features'][column] = feature_results
        
        # Overall drift detection
        drift_ratio = (drift_results['summary']['features_with_drift'] / 
                      drift_results['summary']['total_features'])
        drift_results['overall_drift'] = drift_ratio > 0.3  # 30% of features drifting
        
        return drift_results