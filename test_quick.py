#!/usr/bin/env python3
"""
Quick test script - runs complete pipeline with smaller dataset for demonstration
"""

from main import InsiderThreatDetectionSystem

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AI-DRIVEN INSIDER THREAT DETECTION - QUICK TEST")
    print("="*80 + "\n")
    
    # Initialize system
    system = InsiderThreatDetectionSystem(verbose=True)
    
    # Run with small dataset for testing
    results = system.run_complete_pipeline(n_users=50, n_days=50)
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print("="*80)
    print("\nKey Results:")
    print(f"  Baseline F1-Score: {results['baseline_metrics']['f1']:.4f}")
    print(f"  Fused F1-Score: {results['fusion_metrics']['f1']:.4f}")
    print(f"  Evasion Degradation: {(1 - results['evasion_results']['detection_rate'].mean()) * 100:.1f}%")
    print(f"  Execution Time: {results['duration']:.1f} seconds")
    
    print("\n[+] Full system working successfully!")
