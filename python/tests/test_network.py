"""
Tests for DSSTNE Network class.
"""

import pytest
import numpy as np

from dsstne.network import Network, NetworkConfig, TopKResult


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NetworkConfig()
        
        assert config.batch_size == 32
        assert config.max_k == -1
        assert config.normalize_output is False
        assert config.gpu_id == -1
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = NetworkConfig(
            batch_size=64,
            max_k=100,
            normalize_output=True,
            gpu_id=0
        )
        
        assert config.batch_size == 64
        assert config.max_k == 100
        assert config.normalize_output is True
        assert config.gpu_id == 0
        
    def test_validate_batch_size(self):
        """Test batch_size validation."""
        config = NetworkConfig(batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
            
        config = NetworkConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()
            
    def test_validate_max_k(self):
        """Test max_k validation."""
        config = NetworkConfig(max_k=-2)
        
        with pytest.raises(ValueError, match="max_k must be -1 or positive"):
            config.validate()
            
    def test_valid_config_passes_validation(self):
        """Test that valid config passes validation."""
        config = NetworkConfig(batch_size=32, max_k=10)
        config.validate()  # Should not raise


class TestTopKResult:
    """Tests for TopKResult dataclass."""
    
    def test_creation(self):
        """Test TopKResult creation."""
        indices = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
        scores = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)
        
        result = TopKResult(indices=indices, scores=scores, k=3)
        
        assert result.k == 3
        assert len(result) == 2
        np.testing.assert_array_equal(result.indices, indices)
        np.testing.assert_array_equal(result.scores, scores)
        
    def test_getitem(self):
        """Test accessing individual results."""
        indices = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
        scores = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32)
        
        result = TopKResult(indices=indices, scores=scores, k=3)
        
        idx, sc = result[0]
        np.testing.assert_array_equal(idx, [1, 2, 3])
        np.testing.assert_array_equal(sc, [0.9, 0.8, 0.7])


class TestNetwork:
    """Tests for Network class."""
    
    def test_cannot_create_directly(self):
        """Test that Network cannot be created directly."""
        with pytest.raises(RuntimeError, match="Use Network.load"):
            Network('model.nc')
            
    def test_load_creates_instance(self, mock_model_path):
        """Test that load() creates a Network instance."""
        network = Network.load(mock_model_path, batch_size=32)
        
        assert network is not None
        assert network.batch_size == 32
        network.close()
        
    def test_context_manager(self, mock_model_path):
        """Test using Network as context manager."""
        with Network.load(mock_model_path) as network:
            assert network is not None
            assert not network._is_closed
            
        assert network._is_closed
        
    def test_close_is_idempotent(self, mock_model_path):
        """Test that close() can be called multiple times."""
        network = Network.load(mock_model_path)
        
        network.close()
        assert network._is_closed
        
        # Should not raise
        network.close()
        assert network._is_closed
        
    def test_predict_raises_when_closed(self, mock_model_path):
        """Test that predict raises when network is closed."""
        network = Network.load(mock_model_path)
        network.close()
        
        data = np.random.randn(10, 128).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="Network is closed"):
            network.predict(data)
            
    def test_top_k_raises_when_closed(self, mock_model_path):
        """Test that top_k raises when network is closed."""
        network = Network.load(mock_model_path)
        network.close()
        
        data = np.random.randn(10, 128).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="Network is closed"):
            network.top_k(data, k=10)
            
    def test_top_k_validates_k(self, mock_model_path):
        """Test that top_k validates k parameter."""
        with Network.load(mock_model_path) as network:
            data = np.random.randn(10, 128).astype(np.float32)
            
            with pytest.raises(ValueError, match="k must be positive"):
                network.top_k(data, k=0)
                
            with pytest.raises(ValueError, match="k must be positive"):
                network.top_k(data, k=-5)
                
    def test_predict_returns_array(self, mock_model_path):
        """Test that predict returns a numpy array."""
        with Network.load(mock_model_path, batch_size=32) as network:
            data = np.random.randn(10, 128).astype(np.float32)
            result = network.predict(data)
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 10  # batch size
            
    def test_top_k_returns_result(self, mock_model_path):
        """Test that top_k returns TopKResult."""
        with Network.load(mock_model_path, batch_size=32) as network:
            data = np.random.randn(10, 128).astype(np.float32)
            result = network.top_k(data, k=5)
            
            assert isinstance(result, TopKResult)
            assert result.k == 5
            assert result.indices.shape == (10, 5)
            assert result.scores.shape == (10, 5)
            
    def test_name_property(self, mock_model_path):
        """Test network name property."""
        with Network.load(mock_model_path) as network:
            # Name should be derived from filename
            assert 'test_model' in network.name
            
    def test_predict_with_dict_input(self, mock_model_path):
        """Test predict with dictionary input."""
        with Network.load(mock_model_path) as network:
            inputs = {
                'input': np.random.randn(10, 128).astype(np.float32)
            }
            result = network.predict(inputs)
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 10
            
    def test_validate_input_dimension(self, mock_model_path):
        """Test input dimension validation."""
        with Network.load(mock_model_path) as network:
            # 0-dimensional input should fail
            scalar = np.float32(1.0)
            
            with pytest.raises(ValueError, match="at least 1-dimensional"):
                network.predict(scalar)
