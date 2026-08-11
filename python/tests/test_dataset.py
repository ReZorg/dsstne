"""
Tests for DSSTNE Dataset classes.
"""

import pytest
import numpy as np
import os

from dsstne.dataset import Dataset, DenseDataset, SparseDataset


class TestDenseDataset:
    """Tests for DenseDataset class."""
    
    def test_creation_from_array(self, sample_dense_data):
        """Test creating DenseDataset from numpy array."""
        dataset = DenseDataset(sample_dense_data)
        
        assert dataset.num_samples == 100
        assert dataset.width == 128
        assert dataset.is_sparse is False
        assert len(dataset) == 100
        
    def test_data_is_contiguous_float32(self, sample_dense_data):
        """Test that data is converted to contiguous float32."""
        # Create non-contiguous data
        data = np.random.randn(100, 128).astype(np.float64)
        data = data[::2]  # Non-contiguous
        
        dataset = DenseDataset(data)
        
        assert dataset.data.dtype == np.float32
        assert dataset.data.flags['C_CONTIGUOUS']
        
    def test_requires_2d_array(self):
        """Test that 2D array is required."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            DenseDataset(np.array([1, 2, 3]))
            
        with pytest.raises(ValueError, match="Expected 2D array"):
            DenseDataset(np.random.randn(10, 10, 10))
            
    def test_weights_support(self, sample_dense_data):
        """Test sample weights support."""
        weights = np.random.rand(100).astype(np.float32)
        dataset = DenseDataset(sample_dense_data, weights=weights)
        
        assert dataset.weights is not None
        assert len(dataset.weights) == 100
        
    def test_weights_validation(self, sample_dense_data):
        """Test weights length validation."""
        wrong_weights = np.random.rand(50).astype(np.float32)
        
        with pytest.raises(ValueError, match="Weights length"):
            DenseDataset(sample_dense_data, weights=wrong_weights)
            
    def test_get_batch(self, sample_dense_data):
        """Test batch extraction."""
        dataset = DenseDataset(sample_dense_data)
        
        batch = dataset.get_batch(0, 10)
        
        assert batch.shape == (10, 128)
        np.testing.assert_array_equal(batch, sample_dense_data[:10])
        
    def test_batches_iterator(self, sample_dense_data):
        """Test batch iteration."""
        dataset = DenseDataset(sample_dense_data)
        
        batches = list(dataset.batches(batch_size=32))
        
        assert len(batches) == 4  # 100 / 32 = 3.125, rounded up
        assert batches[0].shape == (32, 128)
        assert batches[-1].shape == (4, 128)  # Last batch has remainder
        
    def test_from_file(self, temp_dir, sample_dense_data):
        """Test loading from file."""
        path = os.path.join(temp_dir, 'test_data.npy')
        np.save(path, sample_dense_data)
        
        dataset = DenseDataset.from_file(path)
        
        assert dataset.num_samples == 100
        assert dataset.width == 128
        np.testing.assert_array_almost_equal(dataset.data, sample_dense_data)
        
    def test_repr(self, sample_dense_data):
        """Test string representation."""
        dataset = DenseDataset(sample_dense_data, name='test')
        
        repr_str = repr(dataset)
        
        assert 'DenseDataset' in repr_str
        assert 'test' in repr_str
        assert '(100, 128)' in repr_str


class TestSparseDataset:
    """Tests for SparseDataset class."""
    
    def test_creation_from_csr(self):
        """Test creating SparseDataset from CSR format."""
        indptr = np.array([0, 2, 5, 7], dtype=np.uint64)
        indices = np.array([0, 2, 1, 3, 4, 0, 2], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        shape = (3, 5)
        
        dataset = SparseDataset(indptr, indices, values, shape)
        
        assert dataset.num_samples == 3
        assert dataset.width == 5
        assert dataset.nnz == 7
        assert dataset.is_sparse is True
        
    def test_from_coo(self, sample_sparse_data):
        """Test creating from COO format."""
        dataset = SparseDataset.from_coo(
            row_indices=sample_sparse_data['row_indices'],
            col_indices=sample_sparse_data['col_indices'],
            values=sample_sparse_data['values'],
            shape=sample_sparse_data['shape']
        )
        
        assert dataset.num_samples == 100
        assert dataset.width == 1000
        assert dataset.nnz == len(sample_sparse_data['values'])
        
    def test_density(self):
        """Test density calculation."""
        indptr = np.array([0, 1, 2, 3], dtype=np.uint64)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        shape = (3, 10)
        
        dataset = SparseDataset(indptr, indices, values, shape)
        
        assert dataset.density == 0.1  # 3 / 30
        
    def test_validation_indptr_length(self):
        """Test indptr length validation."""
        indptr = np.array([0, 2, 5], dtype=np.uint64)  # Wrong length
        indices = np.array([0, 2, 1, 3, 4], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        shape = (3, 5)  # Expects indptr of length 4
        
        with pytest.raises(ValueError, match="indptr length"):
            SparseDataset(indptr, indices, values, shape)
            
    def test_validation_indices_values_length(self):
        """Test indices/values length match validation."""
        indptr = np.array([0, 2, 5, 7], dtype=np.uint64)
        indices = np.array([0, 2, 1, 3, 4, 0], dtype=np.uint32)  # Wrong length
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        shape = (3, 5)
        
        with pytest.raises(ValueError, match="indices length"):
            SparseDataset(indptr, indices, values, shape)
            
    def test_to_dense(self):
        """Test conversion to dense array."""
        indptr = np.array([0, 2, 4], dtype=np.uint64)
        indices = np.array([0, 2, 1, 3], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        shape = (2, 5)
        
        dataset = SparseDataset(indptr, indices, values, shape)
        dense = dataset.to_dense()
        
        expected = np.array([
            [1.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 4.0, 0.0]
        ], dtype=np.float32)
        
        np.testing.assert_array_equal(dense, expected)
        
    def test_get_batch(self):
        """Test batch extraction."""
        indptr = np.array([0, 2, 4, 6], dtype=np.uint64)
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        shape = (3, 10)
        
        dataset = SparseDataset(indptr, indices, values, shape)
        
        batch_indptr, batch_indices, batch_values = dataset.get_batch(0, 2)
        
        # First two rows
        np.testing.assert_array_equal(batch_indptr, [0, 2, 4])
        np.testing.assert_array_equal(batch_indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(batch_values, [1.0, 2.0, 3.0, 4.0])
        
    def test_save_and_load(self, temp_dir):
        """Test save and load roundtrip."""
        indptr = np.array([0, 2, 4], dtype=np.uint64)
        indices = np.array([0, 2, 1, 3], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        shape = (2, 5)
        
        original = SparseDataset(indptr, indices, values, shape)
        
        path = os.path.join(temp_dir, 'sparse.npz')
        original.save(path)
        
        loaded = SparseDataset.from_file(path)
        
        assert loaded.num_samples == original.num_samples
        assert loaded.width == original.width
        assert loaded.nnz == original.nnz
        np.testing.assert_array_equal(loaded.indptr, original.indptr)
        np.testing.assert_array_equal(loaded.indices, original.indices)
        np.testing.assert_array_equal(loaded.values, original.values)
        
    def test_repr(self):
        """Test string representation."""
        indptr = np.array([0, 2, 4], dtype=np.uint64)
        indices = np.array([0, 2, 1, 3], dtype=np.uint32)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        shape = (2, 5)
        
        dataset = SparseDataset(indptr, indices, values, shape, name='test')
        repr_str = repr(dataset)
        
        assert 'SparseDataset' in repr_str
        assert 'test' in repr_str
        assert '(2, 5)' in repr_str
        assert 'nnz=4' in repr_str


class TestSparseDatasetFromScipy:
    """Tests for creating SparseDataset from scipy sparse matrices."""
    
    def test_from_scipy_csr(self):
        """Test creating from scipy CSR matrix."""
        pytest.importorskip('scipy')
        from scipy.sparse import csr_matrix
        
        data = np.array([1, 2, 3, 4, 5, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        indptr = np.array([0, 2, 3, 6])
        
        scipy_sparse = csr_matrix((data, indices, indptr), shape=(3, 3))
        
        dataset = SparseDataset.from_scipy(scipy_sparse)
        
        assert dataset.num_samples == 3
        assert dataset.width == 3
        assert dataset.nnz == 6
        
    def test_from_scipy_coo(self):
        """Test creating from scipy COO matrix."""
        pytest.importorskip('scipy')
        from scipy.sparse import coo_matrix
        
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        
        scipy_sparse = coo_matrix((data, (row, col)), shape=(3, 3))
        
        dataset = SparseDataset.from_scipy(scipy_sparse)
        
        assert dataset.num_samples == 3
        assert dataset.width == 3
        # COO to CSR conversion may reorder, so just check nnz
        assert dataset.nnz == 6
