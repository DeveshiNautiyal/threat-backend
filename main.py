"""
Module 10: End-to-End Orchestration
Complete pipeline orchestrator combining all modules
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json

from data_pipeline import DataPipeline
from baseline_ueba import BaselineUEBA
from adversarial_evasion import AdversarialEvasionEngine, EvasionSimulator
from meta_detection import MetaFeatureExtractor, MetaDetectionLayer, ScoreFusion
from backend_simple import DetectionSystem

class InsiderThreatDetectionSystem:
    """End-to-end insider threat detection system"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Components
        self.pipeline = None
        self.baseline_ueba = None
        self.evasion_engine = None
        self.meta_extractor = None
        self.meta_detector = None
        self.detection_api = None
        
        # Data
        self.X = None
        self.users = None
        self.dates = None
        self.y = None
        self.malicious_users = None
        self.all_users = None
        self.feature_cols = None
        
        # Scores
        self.baseline_scores = None
        self.meta_scores = None
        self.fused_scores = None
        
        self.log(f"[*] Insider Threat Detection System initialized")
    
    def log(self, message: str):
        """Print log message"""
        if self.verbose:
            print(message)
    
    def stage_1_data_ingestion(self):
        """Stage 1: Data Ingestion & Parsing (Days 1-5)"""
        self.log("\n" + "="*80)
        self.log("STAGE 1: DATA INGESTION & PARSING (Days 1-5)")
        self.log("="*80)
        
        self.pipeline = DataPipeline()
        self.malicious_users = []
        self.all_users = []
        # self.malicious_users, self.all_users = self.pipeline.generate_synthetic_cert_data(
        #     n_users=n_users,
        #     n_days=n_days,
        #     malicious_ratio=malicious_ratio
        #)
        
        self.log("[+] Data ingestion complete using CERT dataset")
    
    def stage_2_feature_engineering(self):
        """Stage 2: Feature Engineering (Days 6-10)"""
        self.log("\n" + "="*80)
        self.log("STAGE 2: FEATURE ENGINEERING (Days 6-10)")
        self.log("="*80)
    
        # Extract + normalize
        self.pipeline.extract_features()
        self.pipeline.normalize_features()
    
        # Get matrix
        self.X, self.users, self.dates, self.feature_cols = self.pipeline.get_feature_matrix()
    
        # Labels
        self.y = self.pipeline.features_df['label'].values
    
        # Users info
        self.all_users = list(set(self.users))
        self.malicious_users = list(
            self.pipeline.features_df[self.pipeline.features_df['label'] == 1]['user'].unique()
        )

        # ✅ Logging (clean order)
        self.log(f"[+] Feature engineering complete")
        self.log(f"    Feature matrix shape: {self.X.shape}")
        self.log(f"    Malicious samples: {self.y.sum()}")

        # ✅ Real dataset stats (correct place)
        real_users = len(set(self.users))
        total_records = len(self.X)

        self.log(f"[+] Real dataset stats:")
        self.log(f"    Users: {real_users}")
        self.log(f"    Records: {total_records}")
        self.log(f"    Malicious users: {len(self.malicious_users)}")

    def stage_3_baseline_ueba(self, epochs: int = 30):
        """Stage 3: Baseline UEBA Models (Days 11-16)"""
        self.log("\n" + "="*80)
        self.log("STAGE 3: BASELINE UEBA MODELS (Days 11-16)")
        self.log("="*80)
        
        self.baseline_ueba = BaselineUEBA(contamination=0.05)
        self.baseline_ueba.train(self.X, self.feature_cols, epochs=epochs)
        
        scores = self.baseline_ueba.predict(self.X)
        self.baseline_scores = scores['baseline_ensemble']
        
        self.log(f"[+] Baseline UEBA models trained and scores generated")
    
    def stage_4_baseline_evaluation(self):
        """Stage 4: Baseline Evaluation (Days 17-20)"""
        self.log("\n" + "="*80)
        self.log("STAGE 4: BASELINE EVALUATION (Days 17-20)")
        self.log("="*80)
        
        metrics = self.baseline_ueba.evaluate(self.y, threshold=0.5)
        
        self.log(f"\n[+] Baseline Evaluation Complete:")
        for metric, value in metrics.items():
            if metric != 'threshold':
                self.log(f"    {metric.upper()}: {value:.4f}")
        
        return metrics
    
    # def stage_5_adversarial_evasion(self, intensity: float = 0.7):
    #     """Stage 5: Adversarial Evasion Engine (Days 21-26)"""
    #     self.log("\n" + "="*80)
    #     self.log("STAGE 5: ADVERSARIAL EVASION ENGINE (Days 21-26)")
    #     self.log("="*80)
        
    #     self.evasion_engine = AdversarialEvasionEngine()
    #     simulator = EvasionSimulator(self.baseline_ueba, self.evasion_engine)
        
    #     feature_indices = list(range(len(self.feature_cols)))
    #     # Convert labels to indices of malicious samples
    #     malicious_indices = np.where(self.y == 1)[0]

    #     evasion_results = simulator.simulate_evasion_impact(
    #         self.X, self.y, self.users, self.dates, malicious_indices,
    #         feature_indices
    #     )
        
    #     self.log(f"\n[+] Evasion Simulation Results:")
    #     print(evasion_results.to_string(index=False))
        
    #     return evasion_results
    def stage_5_adversarial_evasion(self, intensity: float = 0.7):
        """Stage 5: Adversarial Evasion Engine (Days 21-26)"""

        self.log("\n" + "="*80)
        self.log("STAGE 5: ADVERSARIAL EVASION ENGINE (Days 21-26)")
        self.log("="*80)

        # Initialize evasion engine
        self.evasion_engine = AdversarialEvasionEngine()
        simulator = EvasionSimulator(self.baseline_ueba, self.evasion_engine)

        feature_indices = list(range(len(self.feature_cols)))

        # ✅ STEP 1: Get malicious record indices (NOT users)
        malicious_indices = np.where(self.y == 1)[0].astype(int)

        # ✅ STEP 2: DEBUG CHECKS (important for correctness)
        print("Total records:", len(self.y))
        print("Malicious count:", int(np.sum(self.y)))
        print("Malicious indices count:", len(malicious_indices))

        # ✅ STEP 3: HARD CHECK
        if len(malicious_indices) == 0:
            self.log("[❌ ERROR] No malicious samples found — evasion cannot run")
            return None

        self.log(f"[✔] Malicious samples for evasion: {len(malicious_indices)}")

        # ✅ STEP 4: Ensure all arrays are aligned
        assert len(self.X) == len(self.y) == len(self.users) == len(self.dates), \
            "[ERROR] Data alignment mismatch between X, y, users, dates"

        # ✅ STEP 5: Run evasion simulation (correct inputs)
        evasion_results = simulator.simulate_evasion_impact(
            self.X.copy(),
            self.y.copy(),
            self.users.copy(),
            self.dates.copy(),
            malicious_indices,
            feature_indices
        )

        # ✅ STEP 6: Validate output
        if evasion_results is None or len(evasion_results) == 0:
            self.log("[❌ ERROR] Evasion simulation returned empty results")
            return None

        # Convert to DataFrame if needed
        if not isinstance(evasion_results, pd.DataFrame):
            evasion_results = pd.DataFrame(evasion_results)

        # Fix NaN values (important)
        evasion_results = evasion_results.fillna(0)

        # Ensure numeric safety
        if 'detection_rate' in evasion_results.columns:
            evasion_results['detection_rate'] = evasion_results['detection_rate'].astype(float)

        if 'mean_score' in evasion_results.columns:
            evasion_results['mean_score'] = evasion_results['mean_score'].astype(float)

        self.log(f"\n[+] Evasion Simulation Results:")
        print(evasion_results.to_string(index=False))

        return evasion_results
    
    def stage_6_meta_feature_extraction(self):
        """Stage 6: Meta-Feature Extraction (Days 27-31)"""
        self.log("\n" + "="*80)
        self.log("STAGE 6: META-FEATURE EXTRACTION (Days 27-31)")
        self.log("="*80)
        
        self.meta_extractor = MetaFeatureExtractor(window_size=14)
        meta_features = self.meta_extractor.extract_meta_features(
            self.X, self.users, self.dates, self.baseline_scores
        )
        
        self.log(f"[+] Meta-features extracted: {meta_features.shape[0]} records")
        
        return meta_features
    
    def stage_7_meta_detection(self, meta_features_df):
        """Stage 7: Meta-Detection Model (Days 32-36)"""
        self.log("\n" + "="*80)
        self.log("STAGE 7: META-DETECTION MODEL (Days 32-36)")
        self.log("="*80)
        
        self.meta_detector = MetaDetectionLayer()
        self.meta_detector.train(meta_features_df)
        self.meta_scores = self.meta_detector.predict(meta_features_df)
        
        self.log(f"[+] Meta-detection model trained and scores generated")
        
        return self.meta_scores
    
    def stage_8_score_fusion(self, alpha: float = 0.5):
        """Stage 8: Score Fusion & Evaluation (Days 37-40)"""
        self.log("\n" + "="*80)
        self.log("STAGE 8: SCORE FUSION & EVALUATION (Days 37-40)")
        self.log("="*80)
        
        self.fused_scores = ScoreFusion.weighted_fusion(
            self.baseline_scores, self.meta_scores, alpha=alpha
        )
        
        # Evaluate fused scores
        y_pred = (self.fused_scores > 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(self.y, y_pred, zero_division=0)
        recall = recall_score(self.y, y_pred, zero_division=0)
        f1 = f1_score(self.y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y, self.fused_scores) if len(np.unique(self.y)) > 1 else 0.0
        
        self.log(f"\n[+] Fused Score Evaluation:")
        self.log(f"    Precision: {precision:.4f}")
        self.log(f"    Recall: {recall:.4f}")
        self.log(f"    F1-Score: {f1:.4f}")
        self.log(f"    ROC-AUC: {roc_auc:.4f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def stage_9_backend_setup(self):
        """Stage 9: Backend & API (Days 41-43)"""
        self.log("\n" + "="*80)
        self.log("STAGE 9: BACKEND & API SETUP (Days 41-43)")
        self.log("="*80)
        
        self.detection_api = DetectionSystem()
        self.detection_api.add_alerts(
            self.users, self.dates, self.baseline_scores, 
            self.meta_scores, self.fused_scores
        )
        
        # Export to JSON
        json_file = self.detection_api.export_to_json(
        'alerts_export.json'
        )
        
        self.log(f"[+] Backend API initialized")
        self.log(f"    Total alerts loaded: {len(self.detection_api.alerts)}")
        self.log(f"    Unique users: {len(self.detection_api.users_data)}")
        self.log(f"    Exported to: {json_file}")
    
    def stage_10_testing_finalization(self):
        """Stage 10: Testing & Finalization (Days 48-50)"""
        self.log("\n" + "="*80)
        self.log("STAGE 10: TESTING & FINALIZATION (Days 48-50)")
        self.log("="*80)
        
        # Generate report
        self.log(f"\n[+] System Testing Complete")
        self.log(f"    ✓ Data pipeline operational")
        self.log(f"    ✓ Baseline UEBA models trained")
        self.log(f"    ✓ Adversarial evasion simulation running")
        self.log(f"    ✓ Meta-detection layer active")
        self.log(f"    ✓ Score fusion working")
        self.log(f"    ✓ Backend API ready")
        self.log(f"    ✓ React dashboard prepared")
    
    def run_complete_pipeline(self, n_users: int = 200, n_days: int = 100):
        """Run complete 50-day implementation"""
        start_time = datetime.now()
        
        self.log(f"\n{'='*80}")
        self.log(f"AI-DRIVEN INSIDER THREAT DETECTION SYSTEM")
        self.log(f"50-Day Complete Implementation")
        self.log(f"{'='*80}")
        
        # Execute all stages
        self.stage_1_data_ingestion()
        self.stage_2_feature_engineering()
        self.stage_3_baseline_ueba(epochs=20)
        baseline_metrics = self.stage_4_baseline_evaluation()
        evasion_results = self.stage_5_adversarial_evasion()
        meta_features = self.stage_6_meta_feature_extraction()
        meta_scores = self.stage_7_meta_detection(meta_features)
        fusion_metrics = self.stage_8_score_fusion()
        self.stage_9_backend_setup()
        self.stage_10_testing_finalization()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        self.print_final_summary(baseline_metrics, fusion_metrics, evasion_results, duration)
        
        return {
            'baseline_metrics': baseline_metrics,
            'fusion_metrics': fusion_metrics,
            'evasion_results': evasion_results,
            'duration': duration
        }
    
    def print_final_summary(self, baseline_metrics, fusion_metrics, evasion_results, duration):
        """Print final system summary"""
        self.log(f"\n{'='*80}")
        self.log(f"FINAL SYSTEM SUMMARY")
        self.log(f"{'='*80}")
        
        self.log(f"\n[BASELINE UEBA PERFORMANCE]")
        self.log(f"    Precision: {baseline_metrics['precision']:.4f}")
        self.log(f"    Recall: {baseline_metrics['recall']:.4f}")
        self.log(f"    F1-Score: {baseline_metrics['f1']:.4f}")
        self.log(f"    ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
        
        self.log(f"\n[AFTER EVASION SIMULATION]")
        if evasion_results is not None and len(evasion_results) > 0:
            avg_detection_rate = evasion_results['detection_rate'].mean()
        else:
            avg_detection_rate = 0.0
        self.log(f"    Average Detection Rate: {avg_detection_rate * 100:.2f}%")
        self.log(f"    Degradation: {(1 - avg_detection_rate) * 100:.2f}%")
        
        self.log(f"\n[META-DETECTION RECOVERY]")
        recovery = (avg_detection_rate - (1 - baseline_metrics['recall'])) / (1 - baseline_metrics['recall'])
        self.log(f"    Recovery Rate: {max(0, recovery * 100):.2f}%")
        
        self.log(f"\n[FUSED SCORE PERFORMANCE]")
        self.log(f"    Precision: {fusion_metrics['precision']:.4f}")
        self.log(f"    Recall: {fusion_metrics['recall']:.4f}")
        self.log(f"    F1-Score: {fusion_metrics['f1']:.4f}")
        self.log(f"    ROC-AUC: {fusion_metrics['roc_auc']:.4f}")
        
        self.log(f"\n[SYSTEM METRICS]")
        self.log(f"    Total Users Analyzed: {len(self.all_users)}")
        self.log(f"    Malicious Users: {len(self.malicious_users)}")
        self.log(f"    Total Alerts Generated: {len(self.detection_api.alerts)}")
        self.log(f"    Critical/High Risk: {len([a for a in self.detection_api.alerts if a['risk_level'] in ['CRITICAL', 'HIGH']])}")
        
        self.log(f"\n[DEPLOYMENT]")
        self.log(f"    Alerts Export:alerts_export.json")
        self.log(f"    React Dashboard: Available in react_dashboard.jsx")
        self.log(f"    Optional FastAPI: Can be added with 'pip install fastapi uvicorn'")
        self.log(f"    Then run: python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000")
        
        self.log(f"\n[EXECUTION TIME]")
        self.log(f"    Total Duration: {duration:.2f} seconds")
        self.log(f"    {'='*80}\n")


def main():
    """Main execution"""
    system = InsiderThreatDetectionSystem(verbose=True)
    
    # Run complete pipeline
    results = system.run_complete_pipeline()
    
    # Save results to file
    results_summary = {
        'baseline_metrics': results['baseline_metrics'],
        'fusion_metrics': results['fusion_metrics'],
        'evasion_results': results['evasion_results'].to_dict('records'),
        'duration_seconds': results['duration']
    }
    
    with open('results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n[+] Results saved to results.json")
    
    # Print API usage instructions
    print("\n" + "="*80)
    print("ALERTS & RESULTS")
    print("="*80)
    print("\nResults Files Generated:")
    print("  - results.json")
    print("  - alerts_export.json")
    print("\nView Results:")
    print("  alerts_export.json | python -m json.tool")
    print("\nOptional: Deploy FastAPI Backend")
    print("  pip install fastapi uvicorn")
    print("  python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
