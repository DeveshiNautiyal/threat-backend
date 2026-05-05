"""
Module 3: Baseline UEBA Models
Isolation Forest, One-Class SVM, and Autoencoder for anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SimpleAutoencoder:
    """Simple Autoencoder using PCA for anomaly detection"""
    
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.pca = None
    
    def fit(self, X: np.ndarray):
        """Fit PCA-based autoencoder"""
        if self.n_components is None:
            self.n_components = max(1, X.shape[1] // 2)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode to latent space"""
        return self.pca.transform(X)
    
    def decode(self, X_encoded: np.ndarray) -> np.ndarray:
        """Decode from latent space"""
        return self.pca.inverse_transform(X_encoded)
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error"""
        X_encoded = self.encode(X)
        X_reconstructed = self.decode(X_encoded)
        return np.mean((X - X_reconstructed) ** 2, axis=1)


class BaselineUEBA:
    """Baseline UEBA system with three anomaly detection models"""
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        
        # Models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
        self.autoencoder = SimpleAutoencoder()
        
        # Scalers
        self.scaler_if = StandardScaler()
        self.scaler_svm = StandardScaler()
        self.scaler_ae = StandardScaler()
        
        # Scores
        self.if_scores = None
        self.svm_scores = None
        self.ae_scores = None
        self.baseline_scores = None
        
        self.feature_cols = None
        
    def train(self, X: np.ndarray, feature_names: list, epochs: int = 50, batch_size: int = 32):
        """Train all three baseline models"""
        print(f"\n[*] Training Baseline UEBA Models")
        print(f"    Input shape: {X.shape}")
        
        self.feature_cols = feature_names
        
        # Isolation Forest
        print("    [*] Training Isolation Forest...")
        X_scaled_if = self.scaler_if.fit_transform(X)
        self.isolation_forest.fit(X_scaled_if)
        print("    [+] Isolation Forest trained")
        
        # One-Class SVM
        print("    [*] Training One-Class SVM...")
        X_scaled_svm = self.scaler_svm.fit_transform(X)
        self.ocsvm.fit(X_scaled_svm)
        print("    [+] One-Class SVM trained")
        
        # Autoencoder
        print("    [*] Training Autoencoder...")
        X_scaled_ae = self.scaler_ae.fit_transform(X)
        self.autoencoder.fit(X_scaled_ae)
        print("    [+] Autoencoder trained")
    
    def predict(self, X: np.ndarray) -> dict:
        """Generate anomaly scores from all models"""
        print(f"\n[*] Generating baseline anomaly scores")
        
        # Isolation Forest scores (convert to 0-1, higher = more anomalous)
        X_scaled_if = self.scaler_if.transform(X)
        if_preds = self.isolation_forest.predict(X_scaled_if)  # -1 for anomaly, 1 for normal
        if_scores_raw = self.isolation_forest.score_samples(X_scaled_if)
        self.if_scores = 1 / (1 + np.exp(if_scores_raw))  # Sigmoid normalization
        
        # One-Class SVM scores
        X_scaled_svm = self.scaler_svm.transform(X)
        svm_preds = self.ocsvm.predict(X_scaled_svm)
        svm_scores_raw = self.ocsvm.decision_function(X_scaled_svm)
        self.svm_scores = 1 / (1 + np.exp(svm_scores_raw))  # Sigmoid normalization
        
        # Autoencoder scores (reconstruction error)
        X_scaled_ae = self.scaler_ae.transform(X)
        ae_errors = self.autoencoder.reconstruction_error(X_scaled_ae)
        
        # Normalize reconstruction errors
        ae_min, ae_max = ae_errors.min(), ae_errors.max()
        if ae_max > ae_min:
            self.ae_scores = (ae_errors - ae_min) / (ae_max - ae_min)
        else:
            self.ae_scores = np.zeros_like(ae_errors)
        
        # Ensemble baseline score (mean of three models)
        self.baseline_scores = (self.if_scores + self.svm_scores + self.ae_scores) / 3
        
        print(f"[+] Baseline scores generated")
        print(f"    Isolation Forest - Mean: {self.if_scores.mean():.4f}, Std: {self.if_scores.std():.4f}")
        print(f"    One-Class SVM - Mean: {self.svm_scores.mean():.4f}, Std: {self.svm_scores.std():.4f}")
        print(f"    Autoencoder - Mean: {self.ae_scores.mean():.4f}, Std: {self.ae_scores.std():.4f}")
        print(f"    Baseline Ensemble - Mean: {self.baseline_scores.mean():.4f}, Std: {self.baseline_scores.std():.4f}")
        
        return {
            'isolation_forest': self.if_scores,
            'ocsvm': self.svm_scores,
            'autoencoder': self.ae_scores,
            'baseline_ensemble': self.baseline_scores
        }
    
    def evaluate(self, y_true: np.ndarray, threshold: float = 0.5) -> dict:
        """Evaluate baseline model performance"""
        y_pred = (self.baseline_scores >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, self.baseline_scores)
        except:
            roc_auc = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'threshold': threshold
        }
        
        print(f"\n[+] Baseline UEBA Evaluation:")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    ROC-AUC: {roc_auc:.4f}")
        
        return metrics
    
    def get_top_anomalies(self, users: np.ndarray, dates: np.ndarray, top_n: int = 20) -> pd.DataFrame:
        """Get top anomalous activities"""
        df = pd.DataFrame({
            'user': users,
            'date': dates,
            'if_score': self.if_scores,
            'svm_score': self.svm_scores,
            'ae_score': self.ae_scores,
            'baseline_score': self.baseline_scores
        })
        
        return df.nlargest(top_n, 'baseline_score')


if __name__ == "__main__":
    from data_pipeline import DataPipeline
    
    # Generate data
    pipeline = DataPipeline()
    malicious_users, all_users = pipeline.generate_synthetic_cert_data(
        n_users=200, n_days=100, malicious_ratio=0.05
    )
    
    features_df = pipeline.extract_features()
    pipeline.normalize_features()
    
    X, users, dates, feature_cols = pipeline.get_feature_matrix()
    
    # Create labels
    y = np.array([1 if user in malicious_users else 0 for user in users])
    
    # Train baseline models
    ueba = BaselineUEBA(contamination=0.05)
    ueba.train(X, feature_cols, epochs=30)
    
    # Predict
    scores = ueba.predict(X)
    
    # Evaluate
    metrics = ueba.evaluate(y, threshold=0.5)
    
    # Show top anomalies
    top_anomalies = ueba.get_top_anomalies(users, dates, top_n=10)
    print("\n[+] Top 10 Anomalies:")
    print(top_anomalies[['user', 'date', 'baseline_score']])
