"""
Core Tensor Operations Module.

Date: September 2025
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from numpy.typing import NDArray
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorOperations:
    """Core tensor operations with validation and error handling."""
    
    @staticmethod
    def validate_tensor(
        tensor: NDArray,
        name: str = "tensor",
        min_ndim: Optional[int] = None,
        max_ndim: Optional[int] = None
    ) -> None:
        """Validate tensor with detailed error messages."""
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"{name} must be numpy array, got {type(tensor).__name__}")
        
        if min_ndim is not None and tensor.ndim < min_ndim:
            raise ValueError(f"{name} must have at least {min_ndim} dimensions, got {tensor.ndim}")
        
        if max_ndim is not None and tensor.ndim > max_ndim:
            raise ValueError(f"{name} must have at most {max_ndim} dimensions, got {tensor.ndim}")
    
    @staticmethod
    def safe_reshape(tensor: NDArray, target_shape: Tuple[int, ...], name: str = "tensor") -> NDArray:
        """Safely reshape tensor with validation."""
        TensorOperations.validate_tensor(tensor, name)
        
        current_size = np.prod(tensor.shape)
        target_size = np.prod(target_shape)
        
        if current_size != target_size:
            raise ValueError(
                f"Cannot reshape {name} from {tensor.shape} to {target_shape}. "
                f"Element count mismatch: {current_size} != {target_size}"
            )
        
        return tensor.reshape(target_shape)
    
    @staticmethod
    def compute_svd(
        matrix: NDArray,
        bond_dim: Optional[int] = None,
        full_matrices: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute truncated SVD with optional bond dimension limit."""
        TensorOperations.validate_tensor(matrix, "matrix", min_ndim=2, max_ndim=2)
        
        if matrix.size == 0:
            raise ValueError("Cannot compute SVD of empty matrix")
        
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=full_matrices)
            V = Vt.T
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD computation failed: {e}")
        
        if bond_dim is not None:
            max_rank = min(bond_dim, len(S))
            U = U[:, :max_rank]
            S = S[:max_rank]
            V = V[:, :max_rank]
        
        return U, S, V
    
    @staticmethod
    def factorize_dimensions(d: int, n_factors: int = 3, balance: bool = True) -> List[int]:
        """Factorize dimension into n_factors for tensor reshaping."""
        if d <= 0:
            raise ValueError(f"Dimension must be positive, got {d}")
        
        # Find prime factorization
        factors = []
        temp = d
        for p in range(2, int(np.sqrt(d)) + 1):
            while temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        
        if not factors:
            factors = [d]
        
        # Combine factors to get desired number
        if len(factors) < n_factors:
            factors.extend([1] * (n_factors - len(factors)))
        elif len(factors) > n_factors:
            while len(factors) > n_factors:
                factors.sort()
                factors[0] = factors[0] * factors[1]
                factors.pop(1)
        
        if balance:
            factors.sort()
        
        return factors


# Convenience functions
def safe_reshape(*args, **kwargs):
    return TensorOperations.safe_reshape(*args, **kwargs)

def compute_svd(*args, **kwargs):
    return TensorOperations.compute_svd(*args, **kwargs)

def factorize_dimensions(*args, **kwargs):
    return TensorOperations.factorize_dimensions(*args, **kwargs)
