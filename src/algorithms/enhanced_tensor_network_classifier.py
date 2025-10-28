# enhanced_tensor_network_classifier.py
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker, tensor_train
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

class EnhancedTensorNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Improved quantum-inspired classifier with hybrid approach
    Combines tensor networks with classical preprocessing for better accuracy
    """
    
    def __init__(self, bond_dimension=8, pca_components=None, hybrid_mode=True, 
                 classical_fallback=True, random_state=42):
        self.bond_dimension = bond_dimension
        self.pca_components = pca_components
        self.hybrid_mode = hybrid_mode
        self.classical_fallback = classical_fallback
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.use_quantum_principles = True
        
    def _quantum_inspired_feature_map(self, X):
        """Apply quantum-inspired feature mapping"""
        if self.use_quantum_principles:
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            X_normalized = X / norms
            even_feats = X_normalized[:, ::2]
            odd_feats = X_normalized[:, 1::2]
            num_even = even_feats.shape[1]
            num_odd = odd_feats.shape[1]
            num_phases = min(num_even, num_odd)
            if num_phases > 0:
                phase = np.arctan2(even_feats[:, :num_phases], odd_feats[:, :num_phases])
                sin_phase = np.sin(phase)
                cos_phase = np.cos(phase)
            else:
                sin_phase = np.zeros((X.shape[0], num_even))
                cos_phase = np.zeros((X.shape[0], num_even))
            extra = num_even - num_phases
            if extra > 0:
                sin_phase = np.hstack([sin_phase, np.zeros((X.shape[0], extra))])
                cos_phase = np.hstack([cos_phase, np.zeros((X.shape[0], extra))])
            return np.column_stack([X_normalized, sin_phase, cos_phase])
        return X
    
    def _intelligent_tensorization(self, X):
        """Improved tensorization based on data characteristics"""
        n_samples, n_features = X.shape
        
        # For high-dimensional data, use intelligent reshaping
        if n_features > 100:
            # Find optimal tensor shape using prime factorization
            factors = self._prime_factors(n_features)
            
            # Group factors to create balanced tensor dimensions (2D or 3D)
            if len(factors) >= 4:
                # 3D tensor for very high dimensions
                mid1, mid2 = len(factors) // 3, 2 * len(factors) // 3
                dim1 = np.prod(factors[:mid1])
                dim2 = np.prod(factors[mid1:mid2])
                dim3 = np.prod(factors[mid2:])
                tensor_shape = (n_samples, dim1, dim2, dim3)
            elif len(factors) >= 2:
                # 2D tensor for moderate dimensions
                mid = len(factors) // 2
                dim1 = np.prod(factors[:mid])
                dim2 = np.prod(factors[mid:])
                tensor_shape = (n_samples, dim1, dim2)
            else:
                # Fallback to square-like shape
                dim1 = int(np.sqrt(n_features))
                dim2 = n_features // dim1
                if dim1 * dim2 < n_features:
                    dim2 += 1
                tensor_shape = (n_samples, dim1, dim2)
        else:
            # For low dimensions, use simple 2D approach
            dim1 = int(np.sqrt(n_features))
            dim2 = n_features // dim1
            if dim1 * dim2 < n_features:
                dim2 += 1
            tensor_shape = (n_samples, dim1, dim2)
        
        # Pad data if necessary
        total_elements = np.prod(tensor_shape[1:])
        if total_elements > n_features:
            X_padded = np.zeros((n_samples, total_elements))
            X_padded[:, :n_features] = X
            X = X_padded
            
        return X.reshape(tensor_shape), tensor_shape[1:]
    
    def _prime_factors(self, n):
        """Get prime factors of n for intelligent tensor shaping"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors if factors else [n]
    
    def _advanced_tensor_decomposition(self, tensor):
        """Use multiple decomposition methods for robustness"""
        try:
            # Try Tensor Train decomposition first (often more stable)
            tt_factors = tensor_train(tensor, rank=self.bond_dimension)
            return 'tensor_train', tt_factors
        except:
            try:
                # Fallback to Tucker decomposition
                core, factors = tucker(tensor, 
                                     rank=[min(self.bond_dimension, d) for d in tensor.shape])
                return 'tucker', (core, factors)
            except:
                try:
                    # Fallback to PARAFAC
                    factors = parafac(tensor, rank=min(self.bond_dimension, min(tensor.shape)))
                    return 'parafac', factors
                except:
                    # No decomposition possible
                    return 'none', tensor
    
    def _robust_tensor_similarity(self, test_tensor, class_prototypes):
        """Robust similarity computation using multiple prototypes"""
        similarities = []
        for method, decomp, original_tensor in class_prototypes:
            class_tensor = original_tensor  # Use original to avoid reconstruction errors
            try:
                min_dims = [min(a, b) for a, b in zip(test_tensor.shape, class_tensor.shape)]
                slices_t = tuple([slice(0, d) for d in min_dims])
                test_trunc = test_tensor[slices_t]
                class_trunc = class_tensor[slices_t]
                ndim = len(min_dims)
                sim = np.abs(np.tensordot(test_trunc, class_trunc, axes=(range(ndim), range(ndim))))
                similarities.append(sim)
            except Exception:
                try:
                    flat_test = test_tensor.flatten()
                    flat_class = class_tensor.flatten()
                    min_len = min(len(flat_test), len(flat_class))
                    sim = np.abs(np.dot(flat_test[:min_len], flat_class[:min_len]))
                    similarities.append(sim)
                except:
                    similarities.append(0.0)
        return max(similarities) if similarities else 0.0
    
    def fit(self, X, y):
        """Enhanced training with hybrid approach"""
        np.random.seed(self.random_state)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Optional PCA preprocessing for very high dimensions
        if self.pca_components and X_scaled.shape[1] > self.pca_components:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X_scaled = self.pca.fit_transform(X_scaled)
        else:
            self.pca = None
        
        # Quantum-inspired feature mapping
        X_scaled = self._quantum_inspired_feature_map(X_scaled)
        
        # Convert to tensor format
        X_tensor, self.tensor_dims = self._intelligent_tensorization(X_scaled)
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Enhanced class-specific tensor learning
        self.class_representations_ = {}
        self.decomposition_methods_ = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_data = X_tensor[class_mask]
            
            if len(class_data) > 0:
                # Use multiple tensor prototypes per class for better representation
                if len(class_data) > 5:
                    # Create multiple prototypes using clustering-like approach
                    n_prototypes = min(3, len(class_data) // 2)
                    indices = np.random.choice(len(class_data), n_prototypes, replace=False)
                    prototypes = class_data[indices]
                else:
                    prototypes = class_data
                
                # Store all prototypes and their decompositions
                class_prototypes = []
                for prototype in prototypes:
                    method, decomp = self._advanced_tensor_decomposition(prototype)
                    class_prototypes.append((method, decomp, prototype))
                
                self.class_representations_[class_label] = class_prototypes
        
        # Hybrid mode: also train a simple classical backup
        if self.hybrid_mode:
            self.classical_model = RandomForestClassifier(
                n_estimators=10, max_depth=5, random_state=self.random_state
            )
            self.classical_model.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X):
        """Enhanced prediction with hybrid approach"""
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Quantum-inspired feature mapping
        X_scaled = self._quantum_inspired_feature_map(X_scaled)
        
        X_tensor, _ = self._intelligent_tensorization(X_scaled)
        
        n_samples = X_tensor.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes_))
        
        for i, sample_tensor in enumerate(X_tensor):
            similarities = []
            
            for class_label in self.classes_:
                class_prototypes = self.class_representations_[class_label]
                similarity = self._robust_tensor_similarity(sample_tensor, class_prototypes)
                similarities.append(similarity)
            
            # Normalize to probabilities
            similarities = np.array(similarities)
            if similarities.sum() > 0:
                tensor_probs = similarities / similarities.sum()
            else:
                tensor_probs = np.ones(self.n_classes_) / self.n_classes_
            
            # Hybrid approach: combine with classical predictions
            if self.hybrid_mode:
                classical_probs = self.classical_model.predict_proba(X_scaled[i:i+1])[0]
                # Weighted combination (favor tensor network for speed, classical for accuracy)
                alpha = 0.3  # Weight for tensor network
                probabilities[i] = alpha * tensor_probs + (1 - alpha) * classical_probs
            else:
                probabilities[i] = tensor_probs
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels with fallback mechanism"""
        try:
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        except Exception as e:
            if self.classical_fallback and hasattr(self, 'classical_model'):
                print(f"Tensor prediction failed: {e}. Using classical fallback.")
                X_scaled = self.scaler.transform(X)
                if self.pca is not None:
                    X_scaled = self.pca.transform(X_scaled)
                # Apply quantum map for consistency
                X_scaled = self._quantum_inspired_feature_map(X_scaled)
                return self.classical_model.predict(X_scaled)
            else:
                raise e

def test_enhanced_tnc():
    """Test enhanced TNC across all datasets"""
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_wine
    import os
    
    results = {}
    
    # Test configurations
    configs = [
        {'name': 'Enhanced_TNC_Hybrid', 'hybrid_mode': True, 'pca_components': None},
        {'name': 'Enhanced_TNC_PCA', 'hybrid_mode': True, 'pca_components': 50},
        {'name': 'Enhanced_TNC_Pure', 'hybrid_mode': False, 'pca_components': None}
    ]
    
    datasets = {
        'wine': lambda: load_wine(return_X_y=True),
        'mnist': lambda: (np.load('datasets/mnist.npz')['X'], np.load('datasets/mnist.npz')['y']),
        'synthetic_2000d': lambda: (np.load('datasets/synthetic_2000d.npz')['X'], 
                                   np.load('datasets/synthetic_2000d.npz')['y'])
    }
    
    for dataset_name, load_func in datasets.items():
        print(f"\n=== Testing Enhanced TNC on {dataset_name} ===")
        
        try:
            X, y = load_func()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                                random_state=42, stratify=y)
            
            results[dataset_name] = {}
            
            for config in configs:
                config_name = config['name']
                print(f"\nTesting {config_name}...")
                
                # Enhanced TNC
                start_time = time.time()
                tnc = EnhancedTensorNetworkClassifier(
                    bond_dimension=8,
                    hybrid_mode=config['hybrid_mode'],
                    pca_components=config['pca_components'],
                    random_state=42
                )
                tnc.fit(X_train, y_train)
                tnc_pred = tnc.predict(X_test)
                tnc_time = time.time() - start_time
                tnc_accuracy = accuracy_score(y_test, tnc_pred)
                
                results[dataset_name][config_name] = {
                    'accuracy': tnc_accuracy,
                    'time': tnc_time
                }
                
                print(f"  Accuracy: {tnc_accuracy:.3f}")
                print(f"  Time: {tnc_time:.3f}s")
                
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    results = test_enhanced_tnc()
    print("\n=== ENHANCED TNC RESULTS ===")
    for dataset, dataset_results in results.items():
        if 'error' not in dataset_results:
            print(f"\n{dataset}:")
            for config, metrics in dataset_results.items():
                print(f"  {config}: Acc={metrics['accuracy']:.3f}, Time={metrics['time']:.3f}s")