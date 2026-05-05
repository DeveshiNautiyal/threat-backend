"""
Module 5: Adversarial Evasion Engine
Implements realistic insider evasion strategies to stress-test detection
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import copy

class AdversarialEvasionEngine:
    """Simulates insider evasion strategies"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.evasion_strategies = {
            'activity_splitting': self._activity_splitting,
            'gradual_drift': self._gradual_drift,
            'threshold_hugging': self._threshold_hugging,
            'temporal_obfuscation': self._temporal_obfuscation,
            'account_switching': self._account_switching
        }
    
    def _activity_splitting(self, X: np.ndarray, feature_indices: List[int], 
                           malicious_mask: np.ndarray, intensity: float = 0.7) -> np.ndarray:
        """
        Split malicious activity into smaller pieces to stay below detection threshold
        Reduces peak values but keeps total volume constant
        """
        X_evaded = X.copy()
        
        for idx in np.where(malicious_mask)[0]:
            for feat_idx in feature_indices:
                if X[idx, feat_idx] > 0:
                    # Split activity across more days
                    original_value = X[idx, feat_idx]
                    split_factor = np.random.uniform(2, 5)
                    new_value = original_value / split_factor
                    X_evaded[idx, feat_idx] = new_value * intensity
        
        return X_evaded
    
    def _gradual_drift(self, X: np.ndarray, users: np.ndarray, dates: np.ndarray,
                      feature_indices: List[int], malicious_mask: np.ndarray,
                      intensity: float = 0.7) -> np.ndarray:
        """
        Gradually increase malicious activity over time to avoid sudden spikes
        Mimics natural behavior transition
        """
        X_evaded = X.copy()
        
        # Group by user
        unique_users = np.unique(users)
        
        for user in unique_users:
            user_mask = (users == user) & malicious_mask
            if not user_mask.any():
                continue
            
            user_indices = np.where(user_mask)[0]
            
            # Sort by date
            date_indices = np.argsort(dates[user_indices])
            sorted_indices = user_indices[date_indices]
            
            # Gradually increase anomaly
            n_days = len(sorted_indices)
            for i, idx in enumerate(sorted_indices):
                for feat_idx in feature_indices:
                    if X[idx, feat_idx] > 0:
                        # Gradual increase: low early, high later
                        increase_factor = (i / max(n_days, 1)) * intensity
                        X_evaded[idx, feat_idx] = X[idx, feat_idx] * (1 + increase_factor)
        
        return X_evaded
    
    def _threshold_hugging(self, X: np.ndarray, feature_indices: List[int],
                          malicious_mask: np.ndarray, percentile: float = 75,
                          intensity: float = 0.7) -> np.ndarray:
        """
        Keep anomalous features just below detection threshold (percentile)
        """
        X_evaded = X.copy()
        
        for feat_idx in feature_indices:
            threshold = np.percentile(X[:, feat_idx], percentile)
            
            for idx in np.where(malicious_mask)[0]:
                if X[idx, feat_idx] > threshold:
                    # Clip to just below threshold
                    X_evaded[idx, feat_idx] = threshold * (0.95 + 0.05 * intensity)
        
        return X_evaded
    
    def _temporal_obfuscation(self, X: np.ndarray, feature_indices: List[int],
                             malicious_mask: np.ndarray, intensity: float = 0.7) -> np.ndarray:
        """
        Add noise to temporal patterns to obscure malicious behavior
        """
        X_evaded = X.copy()
        
        for idx in np.where(malicious_mask)[0]:
            for feat_idx in feature_indices:
                noise = np.random.normal(0, X[idx, feat_idx] * 0.1)
                X_evaded[idx, feat_idx] = max(0, X[idx, feat_idx] + noise * intensity)
        
        return X_evaded
    
    def _account_switching(self, X: np.ndarray, users: np.ndarray,
                          malicious_mask: np.ndarray, feature_indices: List[int],
                          intensity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distribute malicious activity across related accounts
        Simulates account sharing or lateral movement
        """
        X_evaded = X.copy()
        users_evaded = users.copy()
        
        unique_malicious_users = np.unique(users[malicious_mask])
        
        for mal_user in unique_malicious_users:
            user_indices = np.where(users == mal_user)[0]
            
            # Create "related" account
            related_account = f"{mal_user}_clone"
            users_evaded[user_indices] = related_account
            
            # Distribute activity
            for idx in user_indices:
                for feat_idx in feature_indices:
                    if X[idx, feat_idx] > 0:
                        X_evaded[idx, feat_idx] = X[idx, feat_idx] * 0.5 * intensity
        
        return X_evaded, users_evaded
    
    def apply_evasion(self, X: np.ndarray, users: np.ndarray, dates: np.ndarray,
                     malicious_users: List[str], feature_indices: List[int] = None,
                     strategy: str = 'activity_splitting', intensity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply specified evasion strategy to malicious records
        
        Args:
            X: Feature matrix
            users: User identifiers
            dates: Date array
            malicious_users: List of malicious user IDs
            feature_indices: Feature columns to apply evasion to (None = all)
            strategy: Evasion strategy name
            intensity: Intensity of evasion (0-1)
        """
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        malicious_mask = np.array([user in malicious_users for user in users])
        
        print(f"\n[*] Applying evasion strategy: {strategy}")
        print(f"    Malicious records: {malicious_mask.sum()}")
        print(f"    Intensity: {intensity}")
        
        if strategy == 'account_switching':
            X_evaded, users_evaded = self._account_switching(
                X, users, malicious_mask, feature_indices, intensity
            )
            return X_evaded, users_evaded
        elif strategy == 'gradual_drift':
            X_evaded = self._gradual_drift(
                X, users, dates, feature_indices, malicious_mask, intensity
            )
            return X_evaded, users
        else:
            X_evaded = self.evasion_strategies[strategy](
                X, feature_indices, malicious_mask, intensity
            )
            return X_evaded, users
    
    def apply_combined_evasion(self, X: np.ndarray, users: np.ndarray, dates: np.ndarray,
                              malicious_users: List[str], feature_indices: List[int] = None,
                              intensity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Apply combination of evasion strategies"""
        print(f"\n[*] Applying combined evasion strategies")
        
        X_evaded = X.copy()
        users_evaded = users.copy()
        
        # Apply activity splitting
        X_temp, _ = self.apply_evasion(
            X_evaded, users_evaded, dates, malicious_users,
            feature_indices, 'activity_splitting', intensity * 0.8
        )
        X_evaded = X_temp
        
        # Apply gradual drift
        X_temp, _ = self.apply_evasion(
            X_evaded, users_evaded, dates, malicious_users,
            feature_indices, 'gradual_drift', intensity * 0.6
        )
        X_evaded = X_temp
        
        # Apply threshold hugging
        X_temp, _ = self.apply_evasion(
            X_evaded, users_evaded, dates, malicious_users,
            feature_indices, 'threshold_hugging', intensity * 0.7
        )
        X_evaded = X_temp
        
        print(f"[+] Combined evasion applied")
        
        return X_evaded, users_evaded


class EvasionSimulator:
    """Simulates detection degradation under evasion"""
    
    def __init__(self, baseline_model, evasion_engine: AdversarialEvasionEngine):
        self.baseline_model = baseline_model
        self.evasion_engine = evasion_engine
        self.results = {}
    
    def simulate_evasion_impact(self, X: np.ndarray, y: np.ndarray, users: np.ndarray,
                               dates: np.ndarray, malicious_users: List[str],
                               feature_indices: List[int] = None,
                               strategies: List[str] = None) -> pd.DataFrame:
        """
        Simulate detection degradation for each evasion strategy
        """
        if strategies is None:
            strategies = ['activity_splitting', 'gradual_drift', 'threshold_hugging',
                         'temporal_obfuscation']
        
        results = []
        
        print(f"\n[*] Simulating evasion impact on baseline detection")
        print(f"    Original detection rate: {(self.baseline_model.baseline_scores > 0.5).sum() / len(y) * 100:.2f}%")
        
        for strategy in strategies:
            print(f"\n[*] Testing strategy: {strategy}")
            
            X_evaded, users_evaded = self.evasion_engine.apply_evasion(
                X, users, dates, malicious_users, feature_indices, strategy, intensity=0.7
            )
            
            # Predict on evaded data
            scores = self.baseline_model.predict(X_evaded)
            
            # Evaluate
            y_pred = (scores['baseline_ensemble'] > 0.5).astype(int)
            
            # Calculate detection rate on malicious records
            malicious_mask = np.array([user in malicious_users for user in users_evaded])
            detected_malicious = (y_pred[malicious_mask] == 1).sum()
            total_malicious = malicious_mask.sum()
            detection_rate = detected_malicious / max(total_malicious, 1)
            
            results.append({
                'strategy': strategy,
                'detection_rate': detection_rate,
                'detected_count': detected_malicious,
                'total_malicious': total_malicious,
                'mean_score': scores['baseline_ensemble'][malicious_mask].mean()
            })
            
            print(f"    Detection rate after evasion: {detection_rate * 100:.2f}%")
            print(f"    Degradation: {(1 - detection_rate) * 100:.2f}%")
        
        return pd.DataFrame(results)


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
    ueba.predict(X)
    
    # Simulate evasion
    evasion_engine = AdversarialEvasionEngine()
    simulator = EvasionSimulator(ueba, evasion_engine)
    
    feature_indices = [i for i in range(len(feature_cols))]
    results_df = simulator.simulate_evasion_impact(
        X, y, users, dates, malicious_users, feature_indices
    )
    
    print("\n[+] Evasion Impact Summary:")
    print(results_df)
