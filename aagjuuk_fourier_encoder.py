from typing import Tuple, Union
import numpy as np

class MultiScaleFourierEncoder:
    """High-Frequency Spatial Position Encoder for Anisotropic Boundary Fields.

    Leverages random Fourier feature mappings to project low-dimensional coordinates 
    into high-dimensional sinusoidal manifolds, mitigating neural network spectral bias 
    and resolving microscopic structural shear boundaries.

    Attributes:
        input_dim (int): Dimensionality of raw input coordinates (typically x, y, z = 3).
        mapping_size (int): Number of projecting sinusoidal feature pairs.
        scale (float): Standard deviation of the Gaussian distribution for projection scales.
        projection_matrix (np.ndarray): Frozen randomized projection mapping tensor.
    """

    def __init__(self, input_dim: int = 3, mapping_size: int = 32, scale: float = 10.0) -> None:
        """Initializes the Fourier Encoder with a randomized Gaussian projection matrix."""
        self.input_dim: int = input_dim
        self.mapping_size: int = mapping_size
        self.scale: float = scale
        
        # Seed initialized for deterministic production runtime behavior
        rng = np.random.default_rng(seed=42)
        self.projection_matrix: np.ndarray = rng.normal(
            loc=0.0, 
            scale=self.scale, 
            size=(self.input_dim, self.mapping_size)
        )

    def encode(self, coords: np.ndarray) -> np.ndarray:
        """Maps continuous coordinate tensors into Fourier feature space.

        Args:
            coords (np.ndarray): An (N, input_dim) spatial coordinate matrix.

        Returns:
            np.ndarray: An (N, 2 * mapping_size) high-frequency coordinate embedding.
            
        Raises:
            ValueError: If coordinate input dimensions do not match initialized input_dim.
        """
        if coords.ndim != 2 or coords.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected coordinates of shape (N, {self.input_dim}), "
                f"received shape {coords.shape} instead."
            )
            
        # Project spatial points into scaled radian space: 2 * pi * X * B
        projected_matrix: np.ndarray = np.dot(coords, self.projection_matrix) * (2.0 * np.pi)
        
        # Calculate sinusoidal bases
        cos_components: np.ndarray = np.cos(projected_matrix)
        sin_components: np.ndarray = np.sin(projected_matrix)
        
        # Concatenate features horizontally to form the final high-dimensional embedding
        return np.hstack((cos_components, sin_components))
