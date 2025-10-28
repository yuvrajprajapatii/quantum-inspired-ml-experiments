# quantum_inspired_dimensionality_reduction.py
import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train, tucker
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import time

class QuantumInspiredDimensionalityReduction(BaseEstimator, TransformerMixin):
    """
    Quantum-inspired dimensionality reduction using tensor decomposition
    Mimics quantum entanglement structure for efficient high-dimensional processing
    """
    
    def __init__(self, n_components=50, bond_dimension=4, entanglement_mode='local',
                 compression_ratio=0.1, random_state=42):
        self.n_components = n_components
        self.bond_dimension = bond_dimension
        self.entanglement_mode = entanglement_mode  # 'local', 'global', 'adaptive'
        self.compression_ratio = compression_ratio
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def _create_entanglement_structure(self, X):
        """Create quantum-inspired entanglement structure"""
        n_samples, n_features = X.shape
        
        if self.entanglement_mode == 'local':
            # Local entanglement: nearby features are correlated
            entanglement_groups = []
            group_size = max(4, n_features // self.n_components)
            
            for i in range(0, n_features, group_size):
                group = list(range(i, min(i + group_size, n_features)))
                entanglement_groups.append(group)
                
        elif self.entanglement_mode == 'global':
            # Global entanglement: all features potentially correlated
            # Create overlapping groups
            entanglement_groups = []
            step = max(1, n_features // (2 * self.n_components))
            group_size = max(8, n_features // self.n_components)
            
            for i in range(0, n_features - group_size + 1, step):
                group = list(range(i, i + group_size))
                entanglement_groups.append(group)
                
        else:  # adaptive
            # Adaptive entanglement based on feature correlations
            corr_matrix = np.corrcoef(X.T)
            entanglement_groups = self._adaptive_grouping(corr_matrix)
        
        return entanglement_groups
    
    def _adaptive_grouping(self, corr_matrix):
        """Create adaptive feature groups based on correlations"""
        n_features = corr_matrix.shape[0]
        groups = []
        used_features = set()
        
        # Start with highest correlation features
        for i in range(n_features):
            if i in used_features:
                continue
                
            # Find most correlated features to i
            correlations = np.abs(corr_matrix[i])
            corr_indices = np.argsort(correlations)[::-1]
            
            group = []
            for idx in corr_indices:
                if len(group) >= max(4, n_features // self.n_components):
                    break
                if idx not in used_features:
                    group.append(idx)
                    used_features.add(idx)
            
            if group:
                groups.append(group)
        
        return groups
    
    def _tensor_compress_group(self, group_data):
        """Compress a feature group using tensor decomposition"""
        if group_data.shape[1] <= 2:
            return group_data  # No compression needed for small groups
        
        # Reshape to tensor format
        n_samples, n_features = group_data.shape
        
        # Create approximate cubic tensor for decomposition
        if n_features >= 8:
            # 3D tensor
            dim1 = int(np.cbrt(n_features))
            dim2 = int(np.sqrt(n_features // dim1))
            dim3 = n_features // (dim1 * dim2)
            
            if dim1 * dim2 * dim3 < n_features:
                # Pad data
                padding = dim1 * dim2 * dim3 - n_features
                group_data_padded = np.hstack([group_data, np.zeros((n_samples, padding))])
            else:
                group_data_padded = group_data[:, :dim1*dim2*dim3]
            
            tensor_data = group_data_padded.reshape(n_samples, dim1, dim2, dim3)
            tensor_shape = (dim1, dim2, dim3)
        else:
            # 2D tensor
            dim1 = int(np.sqrt(n_features))
            dim2 = n_features // dim1
            if dim1 * dim2 < n_features:
                dim2 += 1
                padding = dim1 * dim2 - n_features
                group_data_padded = np.hstack([group_data, np.zeros((n_samples, padding))])
            else:
                group_data_padded = group_data
            
            tensor_data = group_data_padded.reshape(n_samples, dim1, dim2)
            tensor_shape = (dim1, dim2)
        
        # Perform tensor decomposition on average tensor
        avg_tensor = np.mean(tensor_data, axis=0)
        
        try:
            if len(tensor_shape) == 3:
                # Tucker decomposition for 3D
                core, factors = tucker(avg_tensor, 
                                     rank=[min(self.bond_dimension, d) for d in tensor_shape])
                self.group_decompositions_.append(('tucker', (core, factors), tensor_shape))
            else:
                # SVD-based compression for 2D
                U, s, Vt = np.linalg.svd(avg_tensor, full_matrices=False)
                k = min(self.bond_dimension, len(s))
                compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
                self.group_decompositions_.append(('svd', (U[:, :k], s[:k], Vt[:k, :]), tensor_shape))
            
            # Project all samples using the learned compression
            compressed_features = []
            for sample_tensor in tensor_data:
                if len(tensor_shape) == 3:
                    # Project using Tucker factors
                    projected = np.tensordot(sample_tensor, factors[0], axes=([0], [0]))
                    projected = np.tensordot(projected, factors[1], axes=([0], [0]))
                    projected = np.tensordot(projected, factors[2], axes=([0], [0]))
                else:
                    # Project using SVD factors
                    projected = U[:, :k].T @ sample_tensor @ Vt[:k, :].T
                
                compressed_features.append(projected.flatten())
            
            return np.array(compressed_features)
            
        except:
            # Fallback to PCA-like compression
            n_components_group = min(self.bond_dimension, group_data.shape[1])
            U, s, Vt = np.linalg.svd(group_data.T, full_matrices=False)
            compressed = group_data @ U[:, :n_components_group]
            self.group_decompositions_.append(('pca_fallback', U[:, :n_components_group], group_data.shape))
            return compressed
    
    def fit(self, X, y=None):
        """Fit the quantum-inspired dimensionality reduction"""
        np.random.seed(self.random_state)
        
        # Standardize input
        X_scaled = self.scaler.fit_transform(X)
        
        # Create entanglement structure
        self.entanglement_groups_ = self._create_entanglement_structure(X_scaled)
        self.group_decompositions_ = []
        
        print(f"Created {len(self.entanglement_groups_)} entanglement groups")
        
        # Learn tensor compressions for each group
        self.group_transforms_ = []
        
        for i, group in enumerate(self.entanglement_groups_):
            group_data = X_scaled[:, group]
            compressed_group = self._tensor_compress_group(group_data)
            self.group_transforms_.append(compressed_group.shape[1])
            
        # Calculate total output dimensions
        self.output_dim_ = sum(self.group_transforms_)
        
        return self
    
    def transform(self, X):
        """Transform data using learned quantum-inspired compression"""
        X_scaled = self.scaler.transform(X)
        
        compressed_features = []
        
        for i, group in enumerate(self.entanglement_groups_):
            group_data = X_scaled[:, group]
            
            # Apply the learned compression
            method, decomp_data, original_shape = self.group_decompositions_[i]
            
            if method == 'tucker':
                core, factors = decomp_data
                # Project using Tucker decomposition
                projected_group = []
                for sample in group_data:
                    if len(original_shape) == 3:
                        sample_tensor = sample.reshape(original_shape)
                        projected = np.tensordot(sample_tensor, factors[0], axes=([0], [0]))
                        projected = np.tensordot(projected, factors[1], axes=([0], [0]))  
                        projected = np.tensordot(projected, factors[2], axes=([0], [0]))
                        projected_group.append(projected.flatten())
                    else:
                        sample_tensor = sample.reshape(original_shape)
                        projected = factors[0].T @ sample_tensor @ factors[1]
                        projected_group.append(projected.flatten())
                compressed_features.append(np.array(projected_group))
                
            elif method == 'svd':
                U, s, Vt = decomp_data
                if len(original_shape) == 2:
                    projected_group = []
                    for sample in group_data:
                        sample_tensor = sample.reshape(original_shape)
                        projected = U.T @ sample_tensor @ Vt.T
                        projected_group.append(projected.flatten())
                    compressed_features.append(np.array(projected_group))
                    
            else:  # pca_fallback
                transform_matrix = decomp_data
                compressed_features.append(group_data @ transform_matrix)
        
        # Concatenate all compressed groups
        if compressed_features:
            return np.hstack(compressed_features)
        else:
            return X_scaled[:, :self.n_components]  # Fallback

def test_quantum_inspired_dr():
    """Test QIDR on high-dimensional datasets"""
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    datasets = {
        'wine': lambda: load_wine(return_X_y=True),
        'mnist': lambda: (np.load('datasets/mnist.npz')['X'], np.load('datasets/mnist.npz')['y']),
        'synthetic_2000d': lambda: (np.load('datasets/synthetic_2000d.npz')['X'], 
                                   np.load('datasets/synthetic_2000d.npz')['y'])
    }
    
    results = {}
    
    for dataset_name, load_func in datasets.items():
        print(f"\n=== Testing QIDR on {dataset_name} ===")
        
        try:
            X, y = load_func()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                                random_state=42, stratify=y)
            
            original_dim = X_train.shape[1]
            target_dim = min(50, original_dim // 2)
            
            print(f"Original dimensions: {original_dim}")
            print(f"Target dimensions: {target_dim}")
            
            # Test different entanglement modes
            modes = ['local', 'global', 'adaptive']
            results[dataset_name] = {}
            
            for mode in modes:
                print(f"\nTesting {mode} entanglement mode...")
                
                # QIDR
                start_time = time.time()
                qidr = QuantumInspiredDimensionalityReduction(
                    n_components=target_dim,
                    bond_dimension=8,
                    entanglement_mode=mode,
                    random_state=42
                )
                
                X_train_qidr = qidr.fit_transform(X_train)
                X_test_qidr = qidr.transform(X_test)
                qidr_fit_time = time.time() - start_time
                
                print(f"  QIDR output shape: {X_train_qidr.shape}")
                
                # Test classification performance after QIDR
                start_time = time.time()
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_train_qidr, y_train)
                qidr_pred = clf.predict(X_test_qidr)
                qidr_total_time = time.time() - start_time + qidr_fit_time
                qidr_accuracy = accuracy_score(y_test, qidr_pred)
                
                # Compare with PCA
                start_time = time.time()
                pca = PCA(n_components=target_dim, random_state=42)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                clf_pca = RandomForestClassifier(n_estimators=50, random_state=42)
                clf_pca.fit(X_train_pca, y_train)
                pca_pred = clf_pca.predict(X_test_pca)
                pca_time = time.time() - start_time
                pca_accuracy = accuracy_score(y_test, pca_pred)
                
                results[dataset_name][mode] = {
                    'qidr_accuracy': qidr_accuracy,
                    'qidr_time': qidr_total_time,
                    'qidr_output_dim': X_train_qidr.shape[1],
                    'pca_accuracy': pca_accuracy,
                    'pca_time': pca_time,
                    'compression_ratio': X_train_qidr.shape[1] / original_dim,
                    'accuracy_ratio': qidr_accuracy / pca_accuracy if pca_accuracy > 0 else 0
                }
                
                print(f"  QIDR: Acc={qidr_accuracy:.3f}, Time={qidr_total_time:.3f}s")
                print(f"  PCA:  Acc={pca_accuracy:.3f}, Time={pca_time:.3f}s")
                print(f"  Compression: {results[dataset_name][mode]['compression_ratio']:.3f}")
                print(f"  Accuracy ratio: {results[dataset_name][mode]['accuracy_ratio']:.3f}")
                
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
            results[dataset_name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    results = test_quantum_inspired_dr()
    print("\n=== QIDR FINAL RESULTS ===")
    for dataset, modes in results.items():
        if 'error' not in modes:
            print(f"\n{dataset}:")
            for mode, metrics in modes.items():
                print(f"  {mode}: QIDR_Acc={metrics['qidr_accuracy']:.3f}, "
                      f"PCA_Acc={metrics['pca_accuracy']:.3f}, "
                      f"Ratio={metrics['accuracy_ratio']:.3f}")
