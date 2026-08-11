"""
Amazon DSSTNE Python Package
============================

DSSTNE (Deep Scalable Sparse Tensor Network Engine) is a deep learning library
optimized for sparse data and recommendation systems.

This package provides a Pythonic interface to DSSTNE's neural network and
KNN functionality.

Basic Usage
-----------

```python
import dsstne

# Load a trained network
network = dsstne.Network.load('model.nc', batch_size=32)

# Run inference
predictions = network.predict(input_data)

# Get top-K recommendations
top_k = network.top_k(input_data, k=10)
```

For more information, see the documentation at:
https://github.com/amzn/amazon-dsstne
"""

__version__ = '2.0.0'
__author__ = 'Amazon.com, Inc.'

# Import core classes when C extension is available
try:
    from ._dsstne import (
        # Core module functions
        initialize,
        shutdown,
        get_gpu_count,
        get_gpu_memory,
        
        # Low-level network operations
        load_network_netcdf,
        predict_batch,
        calculate_top_k,
    )
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    
    def _not_available(*args, **kwargs):
        raise RuntimeError(
            "DSSTNE C extension not available. "
            "Please build and install the extension module."
        )
    
    initialize = _not_available
    shutdown = _not_available
    get_gpu_count = _not_available
    get_gpu_memory = _not_available
    load_network_netcdf = _not_available
    predict_batch = _not_available
    calculate_top_k = _not_available

# Import high-level API
from .network import Network, NetworkConfig
from .dataset import Dataset, SparseDataset, DenseDataset

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Status
    'is_available',
    
    # Core classes
    'Network',
    'NetworkConfig',
    'Dataset',
    'SparseDataset',
    'DenseDataset',
    
    # Low-level functions (when available)
    'initialize',
    'shutdown',
    'get_gpu_count',
    'get_gpu_memory',
]


def is_available() -> bool:
    """Check if DSSTNE C extension is available."""
    return _HAS_EXTENSION


def cuda_is_available() -> bool:
    """Check if CUDA is available for GPU operations."""
    if not _HAS_EXTENSION:
        return False
    try:
        return get_gpu_count() > 0
    except Exception:
        return False
