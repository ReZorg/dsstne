"""
Dataset classes for DSSTNE.

This module provides classes for creating and managing datasets
compatible with DSSTNE's neural network operations.
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple, Iterator
from abc import ABC, abstractmethod
import numpy as np


class Dataset(ABC):
    """Abstract base class for DSSTNE datasets.
    
    Datasets provide a uniform interface for feeding data to
    DSSTNE neural networks, supporting both dense and sparse
    data formats.
    """
    
    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Get number of samples in the dataset."""
        pass
        
    @property
    @abstractmethod
    def width(self) -> int:
        """Get width (feature dimension) of the dataset."""
        pass
        
    @property
    @abstractmethod
    def is_sparse(self) -> bool:
        """Check if dataset is sparse."""
        pass
        
    @abstractmethod
    def get_batch(self, start: int, end: int) -> np.ndarray:
        """Get a batch of samples.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            
        Returns:
            Array of samples
        """
        pass
        
    def __len__(self) -> int:
        return self.num_samples
        
    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        """Iterate over batches.
        
        Args:
            batch_size: Number of samples per batch
            
        Yields:
            Batches of samples
        """
        for start in range(0, self.num_samples, batch_size):
            end = min(start + batch_size, self.num_samples)
            yield self.get_batch(start, end)


class DenseDataset(Dataset):
    """Dense dataset for DSSTNE.
    
    Stores data as a dense numpy array.
    
    Example:
        ```python
        # From numpy array
        data = np.random.randn(1000, 128).astype(np.float32)
        dataset = DenseDataset(data)
        
        # From file
        dataset = DenseDataset.from_file('data.npy')
        ```
    """
    
    def __init__(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        name: str = "dense_dataset"
    ):
        """Initialize dense dataset.
        
        Args:
            data: Dense data array of shape (num_samples, width)
            weights: Optional sample weights of shape (num_samples,)
            name: Dataset name
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
            
        self._data = np.ascontiguousarray(data, dtype=np.float32)
        self._weights = None
        self._name = name
        
        if weights is not None:
            if weights.shape[0] != data.shape[0]:
                raise ValueError(
                    f"Weights length {weights.shape[0]} doesn't match "
                    f"data length {data.shape[0]}"
                )
            self._weights = np.ascontiguousarray(weights, dtype=np.float32)
            
    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None) -> 'DenseDataset':
        """Load dense dataset from file.
        
        Supports .npy and .npz formats.
        
        Args:
            path: Path to data file
            name: Optional dataset name
            
        Returns:
            Loaded dataset
        """
        if path.endswith('.npz'):
            npz = np.load(path)
            data = npz['data']
            weights = npz.get('weights', None)
        else:
            data = np.load(path)
            weights = None
            
        return cls(
            data,
            weights=weights,
            name=name or path.split('/')[-1]
        )
        
    @property
    def num_samples(self) -> int:
        return self._data.shape[0]
        
    @property
    def width(self) -> int:
        return self._data.shape[1]
        
    @property
    def is_sparse(self) -> bool:
        return False
        
    @property
    def data(self) -> np.ndarray:
        """Get underlying data array."""
        return self._data
        
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get sample weights."""
        return self._weights
        
    def get_batch(self, start: int, end: int) -> np.ndarray:
        return self._data[start:end].copy()
        
    def __repr__(self) -> str:
        return (
            f"DenseDataset(name='{self._name}', "
            f"shape={self._data.shape}, "
            f"dtype={self._data.dtype})"
        )


class SparseDataset(Dataset):
    """Sparse dataset for DSSTNE.
    
    Stores data in CSR (Compressed Sparse Row) format,
    optimized for DSSTNE's sparse GPU operations.
    
    Example:
        ```python
        # From COO format
        row_indices = np.array([0, 0, 1, 2, 2, 2])
        col_indices = np.array([1, 3, 2, 0, 1, 4])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dataset = SparseDataset.from_coo(
            row_indices, col_indices, values,
            shape=(3, 5)
        )
        
        # From scipy sparse matrix
        from scipy.sparse import csr_matrix
        sparse_matrix = csr_matrix(...)
        dataset = SparseDataset.from_scipy(sparse_matrix)
        ```
    """
    
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int],
        weights: Optional[np.ndarray] = None,
        name: str = "sparse_dataset"
    ):
        """Initialize sparse dataset from CSR format.
        
        Args:
            indptr: Row pointer array of shape (num_samples + 1,)
            indices: Column indices of shape (nnz,)
            values: Non-zero values of shape (nnz,)
            shape: (num_samples, width)
            weights: Optional sample weights
            name: Dataset name
        """
        self._indptr = np.ascontiguousarray(indptr, dtype=np.uint64)
        self._indices = np.ascontiguousarray(indices, dtype=np.uint32)
        self._values = np.ascontiguousarray(values, dtype=np.float32)
        self._shape = shape
        self._name = name
        
        if weights is not None:
            if weights.shape[0] != shape[0]:
                raise ValueError(
                    f"Weights length {weights.shape[0]} doesn't match "
                    f"num_samples {shape[0]}"
                )
            self._weights = np.ascontiguousarray(weights, dtype=np.float32)
        else:
            self._weights = None
            
        self._validate()
        
    def _validate(self) -> None:
        """Validate CSR format consistency."""
        if len(self._indptr) != self._shape[0] + 1:
            raise ValueError(
                f"indptr length {len(self._indptr)} doesn't match "
                f"num_samples + 1 = {self._shape[0] + 1}"
            )
        if len(self._indices) != len(self._values):
            raise ValueError(
                f"indices length {len(self._indices)} doesn't match "
                f"values length {len(self._values)}"
            )
        if len(self._indices) != self._indptr[-1]:
            raise ValueError(
                f"nnz mismatch: indices length {len(self._indices)} != "
                f"indptr[-1] {self._indptr[-1]}"
            )
            
    @classmethod
    def from_coo(
        cls,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
        shape: Tuple[int, int],
        weights: Optional[np.ndarray] = None,
        name: str = "sparse_dataset"
    ) -> 'SparseDataset':
        """Create sparse dataset from COO format.
        
        Args:
            row_indices: Row indices of non-zero values
            col_indices: Column indices of non-zero values
            values: Non-zero values
            shape: (num_samples, width)
            weights: Optional sample weights
            name: Dataset name
            
        Returns:
            SparseDataset instance
        """
        # Sort by row index
        sort_order = np.argsort(row_indices)
        row_indices = row_indices[sort_order]
        col_indices = col_indices[sort_order]
        values = values[sort_order]
        
        # Build CSR indptr
        indptr = np.zeros(shape[0] + 1, dtype=np.uint64)
        for row in row_indices:
            indptr[row + 1] += 1
        np.cumsum(indptr, out=indptr)
        
        return cls(
            indptr=indptr,
            indices=col_indices,
            values=values,
            shape=shape,
            weights=weights,
            name=name
        )
        
    @classmethod
    def from_scipy(
        cls,
        sparse_matrix,
        weights: Optional[np.ndarray] = None,
        name: str = "sparse_dataset"
    ) -> 'SparseDataset':
        """Create sparse dataset from scipy sparse matrix.
        
        Args:
            sparse_matrix: scipy.sparse matrix
            weights: Optional sample weights
            name: Dataset name
            
        Returns:
            SparseDataset instance
        """
        # Convert to CSR if not already
        csr = sparse_matrix.tocsr()
        
        return cls(
            indptr=csr.indptr,
            indices=csr.indices,
            values=csr.data,
            shape=csr.shape,
            weights=weights,
            name=name
        )
        
    @classmethod
    def from_file(cls, path: str, name: Optional[str] = None) -> 'SparseDataset':
        """Load sparse dataset from file.
        
        Args:
            path: Path to .npz file
            name: Optional dataset name
            
        Returns:
            Loaded dataset
        """
        npz = np.load(path)
        return cls(
            indptr=npz['indptr'],
            indices=npz['indices'],
            values=npz['values'],
            shape=tuple(npz['shape']),
            weights=npz.get('weights', None),
            name=name or path.split('/')[-1]
        )
        
    @property
    def num_samples(self) -> int:
        return self._shape[0]
        
    @property
    def width(self) -> int:
        return self._shape[1]
        
    @property
    def is_sparse(self) -> bool:
        return True
        
    @property
    def nnz(self) -> int:
        """Get number of non-zero elements."""
        return len(self._values)
        
    @property
    def density(self) -> float:
        """Get density (fraction of non-zero elements)."""
        total = self._shape[0] * self._shape[1]
        return self.nnz / total if total > 0 else 0.0
        
    @property
    def indptr(self) -> np.ndarray:
        """Get CSR row pointers."""
        return self._indptr
        
    @property
    def indices(self) -> np.ndarray:
        """Get CSR column indices."""
        return self._indices
        
    @property
    def values(self) -> np.ndarray:
        """Get CSR values."""
        return self._values
        
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get sample weights."""
        return self._weights
        
    def get_batch(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch in CSR format.
        
        Returns:
            Tuple of (indptr, indices, values) for the batch
        """
        # Extract rows
        start_ptr = self._indptr[start]
        end_ptr = self._indptr[end]
        
        batch_indptr = self._indptr[start:end+1] - start_ptr
        batch_indices = self._indices[start_ptr:end_ptr]
        batch_values = self._values[start_ptr:end_ptr]
        
        return batch_indptr, batch_indices, batch_values
        
    def to_dense(self) -> np.ndarray:
        """Convert to dense array.
        
        Warning: This may require significant memory for large datasets.
        
        Returns:
            Dense numpy array
        """
        dense = np.zeros(self._shape, dtype=np.float32)
        for i in range(self._shape[0]):
            start = self._indptr[i]
            end = self._indptr[i + 1]
            cols = self._indices[start:end]
            vals = self._values[start:end]
            dense[i, cols] = vals
        return dense
        
    def save(self, path: str) -> None:
        """Save sparse dataset to file.
        
        Args:
            path: Path to save (.npz format)
        """
        save_dict = {
            'indptr': self._indptr,
            'indices': self._indices,
            'values': self._values,
            'shape': np.array(self._shape),
        }
        if self._weights is not None:
            save_dict['weights'] = self._weights
        np.savez_compressed(path, **save_dict)
        
    def __repr__(self) -> str:
        return (
            f"SparseDataset(name='{self._name}', "
            f"shape={self._shape}, "
            f"nnz={self.nnz}, "
            f"density={self.density:.4f})"
        )
