"""
Quantum-Inspired ML Experiments Runner
Tests TNC algorithm against baselines on wine dataset.
"""

import os
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import our tensor network classifier
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.algorithms.tensor_network_classifier_v1 import SimpleTensorNetworkClassifier
    HAVE_TNC = True
except ImportError:
    print("Warning: Could not import TensorNetworkClassifier")
    print("Make sure tensor_network_classifier_v1.py is in src/algorithms/")
    HAVE_TNC = False


def run_baseline():
    """Run Random Forest as baseline"""
    print("\n" + "="*50)
    print("BASELINE: Random Forest")
    print("="*50)
    
    # Load data
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    
    # Train model
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Training time: {train_time:.4f}s")
    
    return accuracy, train_time, X_train, X_test, y_train, y_test


def run_tensor_network_classifier(X_train, X_test, y_train, y_test):
    """Run Tensor Network Classifier"""
    print("\n" + "="*50)
    print("QUANTUM-INSPIRED: Tensor Network Classifier")
    print("="*50)
    
    if not HAVE_TNC:
        print("❌ TNC not available - skipping")
        return None, None
    
    try:
        # Create and train TNC
        print("Creating TNC with bond_dim=4...")
        tnc = SimpleTensorNetworkClassifier(bond_dim=4, n_tensor_dims=3)
        
        print(f"Training on {len(X_train)} samples...")
        start = time.time()
        tnc.fit(X_train, y_train)
        train_time = time.time() - start
        
        print(f"Predicting on {len(X_test)} samples...")
        pred = tnc.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"Training time: {train_time:.4f}s")
        print(f"Features: {X_train.shape[1]} → Tensor shape determined automatically")
        
        return accuracy, train_time
        
    except Exception as e:
        print(f"❌ Error running TNC: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    print("\n" + "#"*50)
    print("# QUANTUM-INSPIRED ML EXPERIMENTS")
    print("# Version 1.0 - Clean Implementation")
    print("#"*50)
    
    # Run baseline
    rf_acc, rf_time, X_train, X_test, y_train, y_test = run_baseline()
    
    # Run TNC
    tnc_acc, tnc_time = run_tensor_network_classifier(X_train, X_test, y_train, y_test)
    
    # Compile results
    results = {
        'random_forest': {
            'accuracy': float(rf_acc),
            'training_time': float(rf_time)
        }
    }
    
    if tnc_acc is not None:
        results['tensor_network_classifier'] = {
            'accuracy': float(tnc_acc),
            'training_time': float(tnc_time),
            'speedup_vs_rf': float(rf_time / tnc_time) if tnc_time > 0 else 0
        }
        
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)
        print(f"Random Forest:  {rf_acc:.1%} accuracy in {rf_time:.4f}s")
        print(f"TNC:            {tnc_acc:.1%} accuracy in {tnc_time:.4f}s")
        if tnc_time > 0:
            print(f"Speedup:        {rf_time/tnc_time:.2f}x")
        
        # Analysis
        print("\n" + "="*50)
        print("ANALYSIS")
        print("="*50)
        if tnc_acc >= 0.85:
            print("✅ TNC achieving good accuracy (>85%)")
        elif tnc_acc >= 0.70:
            print("⚠️  TNC achieving moderate accuracy (70-85%)")
        else:
            print("❌ TNC accuracy needs improvement (<70%)")
        
        if tnc_time < rf_time:
            print("✅ TNC is faster than Random Forest")
        else:
            print("⚠️  TNC is slower (expected for v1, needs optimization)")
    
    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    results_file = 'results/metrics/experiment_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
