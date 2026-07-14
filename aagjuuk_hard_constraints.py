import numpy as np

class BoundaryEnforcedSolver:
    """
    Subspace projection solver that mathematically guarantees physical boundary
    conditions are satisfied, removing reliance on soft-penalty physics loss.
    """
    def __init__(self, baseline_solver_func):
        self.raw_solver = baseline_solver_func

    def evaluate_constrained_displacement(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Forces zero-displacement boundary conditions at the wafer chuck boundary (z=0)
        using a smooth, non-linear algebraic distance-scaling function:
        
        u_constrained(x) = u_raw(x) * (1 - exp(-z / lambda))
        """
        raw_displacements = self.raw_solver(coordinates)
        
        # Extract Z-coordinates (assuming shape [N, 3])
        z_coords = coordinates[:, 2:3]
        
        # Characteristic attenuation length (e.g., scale of boundary layer influence)
        lam = 0.05 
        
        # Boundary enforcement vector (guarantees exactly 0.0 displacement at z=0)
        boundary_multiplier = 1.0 - np.exp(-z_coords / lam)
        
        # Element-wise hard constraint projection
        constrained_displacements = raw_displacements * boundary_multiplier
        
        return constrained_displacements
