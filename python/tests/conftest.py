"""
Pytest configuration and fixtures for DSSTNE tests.
"""

import pytest
import numpy as np
from typing import Generator
import tempfile
import os


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_dense_data() -> np.ndarray:
    """Generate sample dense data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def sample_sparse_data():
    """Generate sample sparse data for testing."""
    np.random.seed(42)
    
    num_samples = 100
    width = 1000
    avg_nnz_per_row = 10
    
    row_indices = []
    col_indices = []
    values = []
    
    for i in range(num_samples):
        nnz = np.random.randint(1, avg_nnz_per_row * 2)
        cols = np.random.choice(width, size=nnz, replace=False)
        vals = np.random.randn(nnz).astype(np.float32)
        
        row_indices.extend([i] * nnz)
        col_indices.extend(cols)
        values.extend(vals)
        
    return {
        'row_indices': np.array(row_indices),
        'col_indices': np.array(col_indices),
        'values': np.array(values, dtype=np.float32),
        'shape': (num_samples, width)
    }


@pytest.fixture
def mock_model_path(temp_dir: str) -> str:
    """Create a mock model file path."""
    path = os.path.join(temp_dir, 'test_model.nc')
    # Create an empty file (real tests would need actual model)
    open(path, 'w').close()
    return path


@pytest.fixture(autouse=True)
def reset_numpy_random_state():
    """Reset numpy random state before each test."""
    np.random.seed(42)
    yield


# Skip markers
requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available"
)

requires_extension = pytest.mark.skipif(
    not _extension_available(),
    reason="DSSTNE extension not available"
)


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import dsstne
        return dsstne.cuda_is_available()
    except Exception:
        return False


def _extension_available() -> bool:
    """Check if DSSTNE extension is available."""
    try:
        import dsstne
        return dsstne.is_available()
    except Exception:
        return False
