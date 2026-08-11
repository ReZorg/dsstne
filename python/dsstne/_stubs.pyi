"""
Type stubs for Amazon DSSTNE Python bindings.

This module provides type hints for IDE support and static type checking.
"""

from typing import List, Tuple, Optional, Dict, Any, Union, Iterator
import numpy as np
from numpy.typing import NDArray


# Type aliases
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
UInt32Array = NDArray[np.uint32]
UInt64Array = NDArray[np.uint64]


class Network:
    """Neural network wrapper for DSSTNE."""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        max_k: int = -1
    ) -> None:
        """
        Initialize a DSSTNE network.
        
        Args:
            model_path: Path to the NetCDF model file.
            batch_size: Batch size for inference.
            max_k: Maximum K for top-K predictions (-1 for all).
        """
        ...
    
    @classmethod
    def load(cls, model_path: str, **kwargs: Any) -> "Network":
        """
        Load a network from file.
        
        Args:
            model_path: Path to the model file.
            **kwargs: Additional configuration options.
            
        Returns:
            Loaded Network instance.
        """
        ...
    
    def predict(
        self,
        inputs: Union[FloatArray, Dict[str, FloatArray]],
        k: Optional[int] = None
    ) -> Tuple[IntArray, FloatArray]:
        """
        Run prediction on input data.
        
        Args:
            inputs: Input array or dictionary of named inputs.
            k: Optional override for top-K value.
            
        Returns:
            Tuple of (indices, values) arrays.
        """
        ...
    
    def predict_batch(
        self,
        inputs: Union[FloatArray, Dict[str, FloatArray]],
        batch_size: Optional[int] = None
    ) -> Iterator[Tuple[IntArray, FloatArray]]:
        """
        Run batched prediction with generator output.
        
        Args:
            inputs: Input data.
            batch_size: Optional batch size override.
            
        Yields:
            Tuples of (indices, values) for each batch.
        """
        ...
    
    def top_k(
        self,
        inputs: Union[FloatArray, Dict[str, FloatArray]],
        k: int = 10
    ) -> List[Tuple[IntArray, FloatArray]]:
        """
        Get top-K predictions.
        
        Args:
            inputs: Input data.
            k: Number of top predictions.
            
        Returns:
            List of (indices, values) tuples for each input.
        """
        ...
    
    def get_layer_output(
        self,
        layer_name: str,
        inputs: FloatArray
    ) -> FloatArray:
        """
        Get output of a specific layer.
        
        Args:
            layer_name: Name of the layer.
            inputs: Input data.
            
        Returns:
            Layer output array.
        """
        ...
    
    @property
    def batch_size(self) -> int:
        """Current batch size."""
        ...
    
    @batch_size.setter
    def batch_size(self, value: int) -> None:
        ...
    
    @property
    def layers(self) -> List[str]:
        """List of layer names."""
        ...
    
    @property
    def input_names(self) -> List[str]:
        """List of input layer names."""
        ...
    
    @property
    def output_names(self) -> List[str]:
        """List of output layer names."""
        ...
    
    def close(self) -> None:
        """Release resources."""
        ...
    
    def __enter__(self) -> "Network":
        ...
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        ...


class DenseDataset:
    """Dense dataset for DSSTNE."""
    
    def __init__(
        self,
        data: FloatArray,
        name: str = "input"
    ) -> None:
        """
        Create a dense dataset.
        
        Args:
            data: 2D float array (samples x features).
            name: Dataset name.
        """
        ...
    
    @classmethod
    def from_file(cls, path: str, name: str = "input") -> "DenseDataset":
        """Load dataset from file."""
        ...
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Dataset shape (samples, features)."""
        ...
    
    @property
    def name(self) -> str:
        """Dataset name."""
        ...
    
    def to_numpy(self) -> FloatArray:
        """Convert to NumPy array."""
        ...


class SparseDataset:
    """Sparse dataset for DSSTNE."""
    
    def __init__(
        self,
        indices: UInt32Array,
        values: FloatArray,
        indptr: UInt64Array,
        shape: Tuple[int, int],
        name: str = "input"
    ) -> None:
        """
        Create a sparse dataset in CSR format.
        
        Args:
            indices: Column indices.
            values: Non-zero values.
            indptr: Row pointers.
            shape: Dataset shape (samples, features).
            name: Dataset name.
        """
        ...
    
    @classmethod
    def from_scipy_sparse(
        cls,
        matrix: Any,  # scipy.sparse matrix
        name: str = "input"
    ) -> "SparseDataset":
        """Create from scipy sparse matrix."""
        ...
    
    @classmethod
    def from_file(cls, path: str, name: str = "input") -> "SparseDataset":
        """Load dataset from file."""
        ...
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Dataset shape (samples, features)."""
        ...
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        ...
    
    @property
    def name(self) -> str:
        """Dataset name."""
        ...
    
    def to_dense(self) -> FloatArray:
        """Convert to dense NumPy array."""
        ...


class TopKResult:
    """Result from top-K prediction."""
    
    @property
    def indices(self) -> IntArray:
        """Top-K indices."""
        ...
    
    @property
    def values(self) -> FloatArray:
        """Top-K values/scores."""
        ...
    
    @property
    def k(self) -> int:
        """Number of results."""
        ...
    
    def __len__(self) -> int:
        ...
    
    def __iter__(self) -> Iterator[Tuple[int, float]]:
        ...
    
    def __getitem__(self, idx: int) -> Tuple[int, float]:
        ...


def load_model(
    model_path: str,
    batch_size: int = 32,
    max_k: int = -1
) -> Network:
    """
    Load a DSSTNE model.
    
    Args:
        model_path: Path to the model file.
        batch_size: Batch size for inference.
        max_k: Maximum K for predictions.
        
    Returns:
        Loaded Network instance.
    """
    ...


def generate_netcdf(
    output_path: str,
    data: Union[FloatArray, Dict[str, FloatArray]],
    sparse: bool = False
) -> None:
    """
    Generate NetCDF dataset file.
    
    Args:
        output_path: Output file path.
        data: Data to save.
        sparse: Whether to save as sparse format.
    """
    ...


def get_cuda_device_count() -> int:
    """Get number of available CUDA devices."""
    ...


def get_cuda_device_name(device_id: int = 0) -> str:
    """Get CUDA device name."""
    ...


def get_version() -> str:
    """Get DSSTNE version string."""
    ...


# Constants
VERSION: str
CUDA_VERSION: str
