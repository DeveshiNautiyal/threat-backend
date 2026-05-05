"""
Module 6-7: Meta-Feature Extraction and Meta-Detection Layer
Detects evasion behavior by analyzing behavioral trends over time
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MetaFeatureExtractor:
    """Extracts meta-features that indicate evasion behavior"""
    
    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        self.meta_features_df = None
    
    def compute_threshold_proximity(self, X: np.ndarray, baseline_scores: np.ndarray,
                                   percentile: float = 75) -> np.ndarray:
        """
        Measure how close each record is to detection threshold
        Threshold-hugging behavior: stay just below threshold
        """
        threshold = np.percentile(baseline_scores, percentile)
        proximity = np.zeros(len(X))
        
        for i in range(len(X)):
            if baseline_scores[i] < threshold:
                # Closeness to threshold: 0 = far, 1 = at threshold
                proximity[i] = baseline_scores[i] / max(threshold, 0.001)
            else:
                proximity[i] = 1.0
        
        return proximity
    
    def compute_behavioral_drift(self, X: np.ndarray, users: np.ndarray,
                                dates: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Detect gradual behavioral drift over time
        Measures change in behavior pattern from historical baseline
        """
        if window_size is None:
            window_size = self.window_size
        
        drift = np.zeros(len(X))
        unique_users = np.unique(users)
        
        for user in unique_users:
            user_mask = users == user
            user_indices = np.where(user_mask)[0]
            
            if len(user_indices) < window_size:
                continue
            
            # Sort by date
            date_indices = np.argsort(dates[user_indices])
            sorted_indices = user_indices[date_indices]
            
            # Compute drift for each position
            for i in range(window_size, len(sorted_indices)):
                current_window = X[sorted_indices[i - window_size:i]]
                current_behavior = current_window.mean(axis=0)
                
                historical_window = X[sorted_indices[max(0, i - 2 * window_size):i - window_size]]
                if len(historical_window) > 0:
                    historical_behavior = historical_window.mean(axis=0)
                    
                    # Euclidean distance between current and historical
                    drift[sorted_indices[i]] = np.linalg.norm(current_behavior - historical_behavior)
        
        # Normalize drift
        if drift.max() > 0:
            drift = drift / drift.max()
        
        return drift
    
    def compute_variance_over_time(self, X: np.ndarray, users: np.ndarray,
                                  dates: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Detect increased variance in behavior (inconsistency indicator)
        High variance can indicate evasion attempts (mixing normal and malicious)
        """
        if window_size is None:
            window_size = self.window_size
        
        variance = np.zeros(len(X))
        unique_users = np.unique(users)
        
        for user in unique_users:
            user_mask = users == user
            user_indices = np.where(user_mask)[0]
            
            if len(user_indices) < window_size:
                continue
            
            date_indices = np.argsort(dates[user_indices])
            sorted_indices = user_indices[date_indices]
            
            for i in range(window_size, len(sorted_indices)):
                window = X[sorted_indices[i - window_size:i]]
                variance[sorted_indices[i]] = window.std(axis=0).mean()
        
        # Normalize
        if variance.max() > 0:
            variance = variance / variance.max()
        
        return variance
    
    def compute_consistency_score(self, X: np.ndarray, users: np.ndarray,
                                 dates: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Measure consistency of user behavior
        Sudden changes in consistency can indicate evasion
        """
        if window_size is None:
            window_size = self.window_size
        
        consistency = np.zeros(len(X))
        unique_users = np.unique(users)
        
        for user in unique_users:
            user_mask = users == user
            user_indices = np.where(user_mask)[0]
            
            if len(user_indices) < window_size:
                consistency[user_indices] = 1.0
                continue
            
            date_indices = np.argsort(dates[user_indices])
            sorted_indices = user_indices[date_indices]
            
            # Calculate feature-wise consistency
            for i in range(len(sorted_indices)):
                start_idx = max(0, i - window_size)
                window = X[sorted_indices[start_idx:i + 1]]
                
                if len(window) > 0:
                    # Coefficient of variation per feature
                    cv = np.std(window, axis=0) / (np.mean(np.abs(window), axis=0) + 1e-6)
                    # Inverse: higher CV = lower consistency
                    consistency[sorted_indices[i]] = 1.0 / (1.0 + cv.mean())
        
        return consistency
    
    def extract_meta_features(self, X: np.ndarray, users: np.ndarray, dates: np.ndarray,
                             baseline_scores: np.ndarray) -> pd.DataFrame:
        """
        Extract all meta-features
        """
        print("[*] Extracting meta-features...")
        
        # Compute meta-features
        threshold_proximity = self.compute_threshold_proximity(X, baseline_scores)
        behavioral_drift = self.compute_behavioral_drift(X, users, dates)
        variance_over_time = self.compute_variance_over_time(X, users, dates)
        consistency_score = self.compute_consistency_score(X, users, dates)
        
        # Combine into DataFrame
        self.meta_features_df = pd.DataFrame({
            'user': users,
            'date': dates,
            'threshold_proximity': threshold_proximity,
            'behavioral_drift': behavioral_drift,
            'variance_over_time': variance_over_time,
            'consistency_score': consistency_score
        })
        
        print("[+] Meta-features extracted:")
        print(f"    Threshold Proximity - Mean: {threshold_proximity.mean():.4f}, Std: {threshold_proximity.std():.4f}")
        print(f"    Behavioral Drift - Mean: {behavioral_drift.mean():.4f}, Std: {behavioral_drift.std():.4f}")
        print(f"    Variance Over Time - Mean: {variance_over_time.mean():.4f}, Std: {variance_over_time.std():.4f}")
        print(f"    Consistency Score - Mean: {consistency_score.mean():.4f}, Std: {consistency_score.std():.4f}")
        
        return self.meta_features_df


class MetaDetectionLayer:
    """Detects evasion behavior using meta-features"""
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.meta_scores = None
        self.meta_features_cols = None
    
    def train(self, meta_features_df: pd.DataFrame):
        """Train meta-detection model"""
        print(f"\n[*] Training Meta-Detection Layer")
        
        self.meta_features_cols = [col for col in meta_features_df.columns
                                  if col not in ['user', 'date']]
        
        X_meta = meta_features_df[self.meta_features_cols].values
        
        # Normalize
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        # Train
        self.model.fit(X_meta_scaled)
        
        print(f"[+] Meta-Detection Layer trained")
        print(f"    Features: {self.meta_features_cols}")
    
    def predict(self, meta_features_df: pd.DataFrame) -> np.ndarray:
        """Generate meta-anomaly scores"""
        print(f"\n[*] Generating meta-anomaly scores")
        
        X_meta = meta_features_df[self.meta_features_cols].values
        X_meta_scaled = self.scaler.transform(X_meta)
        
        # Get anomaly scores (convert to 0-1)
        meta_scores_raw = self.model.score_samples(X_meta_scaled)
        self.meta_scores = 1 / (1 + np.exp(meta_scores_raw))  # Sigmoid normalization
        
        print(f"[+] Meta-anomaly scores generated")
        print(f"    Mean: {self.meta_scores.mean():.4f}, Std: {self.meta_scores.std():.4f}")
        
        return self.meta_scores


class ScoreFusion:
    """Fuses baseline and meta-detection scores"""
    
    @staticmethod
    def weighted_fusion(baseline_scores: np.ndarray, meta_scores: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """
        Weighted combination of baseline and meta scores
        alpha = 0.5 gives equal weight to both layers
        """
        fused_scores = alpha * baseline_scores + (1 - alpha) * meta_scores
        return fused_scores
    
    @staticmethod
    def adaptive_fusion(baseline_scores: np.ndarray, meta_scores: np.ndarray) -> np.ndarray:
        """
        Adaptive fusion: weight meta-detection higher if it disagrees with baseline
        Handles cases where baseline misses threats that meta-detection catches
        """
        disagreement = np.abs(baseline_scores - meta_scores)
        
        # If disagreement is high, trust meta-detection more
        alpha = 0.3 + 0.2 * disagreement  # Range: 0.3 to 0.5
        
        fused_scores = alpha * baseline_scores + (1 - alpha) * meta_scores
        return fused_scores
    
    @staticmethod
    def max_fusion(baseline_scores: np.ndarray, meta_scores: np.ndarray) -> np.ndarray:
        """
        Maximum fusion: take highest score from either layer
        Conservative approach: flag if either layer detects anomaly
        """
        return np.maximum(baseline_scores, meta_scores)


if __name__ == "__main__":
    from data_pipeline import DataPipeline
    from baseline_ueba import BaselineUEBA
    
    # Generate data
    pipeline = DataPipeline()
    malicious_users, all_users = pipeline.generate_synthetic_cert_data(
        n_users=200, n_days=100, malicious_ratio=0.05
    )
    
    features_df = pipeline.extract_features()
    pipeline.normalize_features()
    
    X, users, dates, feature_cols = pipeline.get_feature_matrix()
    y = np.array([1 if user in malicious_users else 0 for user in users])
    
    # Train baseline
    print("[*] Training baseline UEBA...")
    ueba = BaselineUEBA()
    ueba.train(X, feature_cols, epochs=20)
    scores = ueba.predict(X)
    
    # Extract meta-features
    extractor = MetaFeatureExtractor()
    meta_features = extractor.extract_meta_features(X, users, dates, scores['baseline_ensemble'])
    
    # Train meta-detection
    meta_detector = MetaDetectionLayer()
    meta_detector.train(meta_features)
    meta_scores = meta_detector.predict(meta_features)
    
    # Fuse scores
    print("\n[*] Score Fusion")
    fused_scores = ScoreFusion.weighted_fusion(scores['baseline_ensemble'], meta_scores, alpha=0.5)
    
    print(f"    Baseline scores - Mean: {scores['baseline_ensemble'].mean():.4f}")
    print(f"    Meta scores - Mean: {meta_scores.mean():.4f}")
    print(f"    Fused scores - Mean: {fused_scores.mean():.4f}")
