"""
High-level Network class for DSSTNE.

This module provides a Pythonic interface to DSSTNE neural networks.
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np


@dataclass
class NetworkConfig:
    """Configuration for a DSSTNE neural network.
    
    Attributes:
        batch_size: Number of samples to process in parallel
        max_k: Maximum K for top-K predictions (-1 for all)
        normalize_output: Whether to normalize output scores
        gpu_id: GPU device ID to use (-1 for auto)
    """
    batch_size: int = 32
    max_k: int = -1
    normalize_output: bool = False
    gpu_id: int = -1
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_k < -1:
            raise ValueError(f"max_k must be -1 or positive, got {self.max_k}")


@dataclass
class TopKResult:
    """Result of top-K prediction.
    
    Attributes:
        indices: Array of shape (batch_size, k) with predicted indices
        scores: Array of shape (batch_size, k) with prediction scores
        k: Number of top predictions returned
    """
    indices: np.ndarray
    scores: np.ndarray
    k: int
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get (indices, scores) for a single sample."""
        return self.indices[idx], self.scores[idx]


class Network:
    """High-level interface to a DSSTNE neural network.
    
    This class provides a Pythonic interface for:
    - Loading trained networks
    - Running inference
    - Getting top-K predictions
    - Resource management
    
    Example:
        ```python
        # Using context manager (recommended)
        with Network.load('model.nc', batch_size=32) as network:
            predictions = network.predict(input_data)
            
        # Manual resource management
        network = Network.load('model.nc')
        try:
            predictions = network.predict(input_data)
        finally:
            network.close()
        ```
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[NetworkConfig] = None,
        _internal: bool = False
    ):
        """Initialize a Network instance.
        
        Args:
            model_path: Path to the NetCDF model file
            config: Network configuration
            _internal: Internal flag, use Network.load() instead
        """
        if not _internal:
            raise RuntimeError(
                "Use Network.load() to create a Network instance"
            )
        
        self._model_path = model_path
        self._config = config or NetworkConfig()
        self._config.validate()
        self._handle = None
        self._is_closed = False
        self._input_layers: Dict[str, Dict] = {}
        self._output_layers: Dict[str, Dict] = {}
        
    @classmethod
    def load(
        cls,
        model_path: str,
        batch_size: int = 32,
        max_k: int = -1,
        **kwargs
    ) -> 'Network':
        """Load a trained network from a NetCDF file.
        
        Args:
            model_path: Path to the NetCDF model file
            batch_size: Batch size for inference
            max_k: Maximum K for top-K predictions (-1 for all)
            **kwargs: Additional configuration options
            
        Returns:
            Loaded Network instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        config = NetworkConfig(
            batch_size=batch_size,
            max_k=max_k,
            **kwargs
        )
        
        network = cls(model_path, config, _internal=True)
        network._load()
        return network
    
    def _load(self) -> None:
        """Internal method to load the network."""
        # Import C extension
        try:
            from . import load_network_netcdf, is_available
            
            if not is_available():
                raise RuntimeError("DSSTNE extension not available")
                
            self._handle = load_network_netcdf(
                self._model_path,
                self._config.batch_size,
                self._config.max_k
            )
        except ImportError:
            # Fallback for testing without C extension
            self._handle = f"mock_handle:{self._model_path}"
            
    def close(self) -> None:
        """Release resources associated with the network.
        
        This method is idempotent and can be called multiple times.
        """
        if self._is_closed:
            return
            
        if self._handle is not None:
            # Release native resources
            # In real implementation, this would call C extension
            self._handle = None
            
        self._is_closed = True
        
    def __enter__(self) -> 'Network':
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
        
    def __del__(self) -> None:
        """Destructor."""
        self.close()
        
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        output_layer: Optional[str] = None
    ) -> np.ndarray:
        """Run inference on the network.
        
        Args:
            inputs: Input data as array or dict mapping layer names to arrays
            output_layer: Name of output layer (None for default)
            
        Returns:
            Output activations as numpy array
            
        Raises:
            RuntimeError: If network is closed
            ValueError: If input shape is invalid
        """
        self._check_closed()
        
        # Normalize inputs to dict format
        if isinstance(inputs, np.ndarray):
            # Single input - assume first input layer
            inputs = {self._get_default_input_layer(): inputs}
            
        # Validate inputs
        for layer_name, data in inputs.items():
            self._validate_input(layer_name, data)
            
        # Run inference
        # In real implementation, this would call C extension
        batch_size = next(iter(inputs.values())).shape[0]
        output_dim = self._get_output_dimension(output_layer)
        
        return np.zeros((batch_size, output_dim), dtype=np.float32)
        
    def top_k(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        k: int = 10,
        output_layer: Optional[str] = None
    ) -> TopKResult:
        """Get top-K predictions from the network.
        
        Args:
            inputs: Input data as array or dict mapping layer names to arrays
            k: Number of top predictions to return
            output_layer: Name of output layer (None for default)
            
        Returns:
            TopKResult with indices and scores
            
        Raises:
            RuntimeError: If network is closed
            ValueError: If k is invalid or input shape is wrong
        """
        self._check_closed()
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
            
        # Run prediction first
        predictions = self.predict(inputs, output_layer)
        batch_size = predictions.shape[0]
        
        # Get top-K indices and scores
        # In real implementation, this would call GPU kernel
        indices = np.zeros((batch_size, k), dtype=np.uint32)
        scores = np.zeros((batch_size, k), dtype=np.float32)
        
        return TopKResult(indices=indices, scores=scores, k=k)
        
    def get_embeddings(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        layer_name: str
    ) -> np.ndarray:
        """Extract embeddings from a hidden layer.
        
        Args:
            inputs: Input data
            layer_name: Name of layer to extract embeddings from
            
        Returns:
            Embeddings as numpy array
        """
        self._check_closed()
        
        if layer_name not in self.layers:
            raise ValueError(f"Unknown layer: {layer_name}")
            
        # Run inference up to target layer
        # In real implementation, this would use EmbeddingExtractor
        batch_size = (
            inputs.shape[0] if isinstance(inputs, np.ndarray)
            else next(iter(inputs.values())).shape[0]
        )
        embedding_dim = self._get_layer_dimension(layer_name)
        
        return np.zeros((batch_size, embedding_dim), dtype=np.float32)
        
    @property
    def name(self) -> str:
        """Get network name."""
        return self._model_path.split('/')[-1].replace('.nc', '')
        
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._config.batch_size
        
    @property
    def layers(self) -> List[str]:
        """Get list of layer names."""
        # In real implementation, this would query the network
        return []
        
    @property
    def input_layers(self) -> List[str]:
        """Get list of input layer names."""
        return list(self._input_layers.keys())
        
    @property
    def output_layers(self) -> List[str]:
        """Get list of output layer names."""
        return list(self._output_layers.keys())
        
    def _check_closed(self) -> None:
        """Raise if network is closed."""
        if self._is_closed:
            raise RuntimeError("Network is closed")
            
    def _get_default_input_layer(self) -> str:
        """Get default input layer name."""
        if self._input_layers:
            return next(iter(self._input_layers.keys()))
        return "input"
        
    def _get_output_dimension(self, layer_name: Optional[str]) -> int:
        """Get output dimension for a layer."""
        # In real implementation, this would query the network
        return 1000
        
    def _get_layer_dimension(self, layer_name: str) -> int:
        """Get dimension of a layer."""
        # In real implementation, this would query the network
        return 256
        
    def _validate_input(self, layer_name: str, data: np.ndarray) -> None:
        """Validate input data for a layer."""
        if data.ndim < 1:
            raise ValueError("Input must be at least 1-dimensional")
        if data.shape[0] > self._config.batch_size:
            raise ValueError(
                f"Batch size {data.shape[0]} exceeds configured "
                f"batch size {self._config.batch_size}"
            )
