# NLFMP
NLFMP

NLFMP: An Efficient Genetic Algorithm Software for the Non-Linear Fitting of Multi-Parameters in Quantum Chemistry

--

## Package Name: `NLFMP`

### Full Project Structure
```
algochem/
├── algochem/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── algorithm.py  # Paper's core algorithm logic
│   │   └── params.py     # Hyperparameter/parameter management
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── ase_calc.py   # ASE Calculator integration
│   │   └── pyscf_interface.py  # PySCF compatibility layer
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py     # Dataset loading for ML models
│   │   └── converter.py  # File format conversion (XYZ, CIF, Gaussian)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py     # Structured logging
│       ├── math_utils.py # Numerical utilities (gradients, symmetries)
│       └── errors.py     # Custom exception handling
├── examples/
│   ├── run_optimization.py  # Reproduce paper's structure optimization
│   ├── run_energy_prediction.py  # Reproduce energy/force benchmarks
│   └── paper_systems/      # Input files from the paper (XYZ/CIF)
├── tests/
│   ├── __init__.py
│   ├── test_core_algorithm.py
│   └── test_ase_calculator.py
├── pyproject.toml  # Modern packaging (Poetry/Pipenv)
├── README.md       # Reproducibility instructions
└── LICENSE         # MIT License (standard for academic software)
```

---

## 1. Core Implementation (`algochem/core/algorithm.py`)
This module contains the **paper’s central algorithm class**. Replace the placeholder logic with the exact mathematical formulation, equations, and steps from your paper (e.g., a new neural network architecture for ML potentials, a modified DFT approximation, or a rare-earth-specific force field).

```python
"""
Core implementation of the algorithm proposed in:
[Insert Paper Title, Authors, Journal, Year]

This module defines the `PaperAlgorithm` class, which encapsulates all key
computational logic (energy calculation, force prediction, gradient computation)
for lanthanide/actinide systems.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from algochem.core.params import AlgorithmParams
from algochem.utils.math_utils import compute_pairwise_distances, numerical_gradients

class PaperAlgorithm:
    """
    Reference implementation of the paper's core algorithm.
    
    Designed for systems containing lanthanide/actinide elements, with support
    for energy, force, and stress calculations (critical for periodic systems).
    
    Attributes:
        params: Algorithm hyperparameters and trainable parameters (if ML-based).
        element_map: Dictionary mapping element symbols to atomic numbers (focused on Ln/An).
        logger: Structured logger for debugging and reproducibility.
    """
    def __init__(
        self,
        params: Optional[Union[Dict, AlgorithmParams]] = None,
        element_list: Optional[List[str]] = None,
        verbose: bool = False
    ):
        # Initialize hyperparameters (from paper)
        self.params = AlgorithmParams(**params) if isinstance(params, Dict) else params or AlgorithmParams()
        
        # Define supported elements (prioritize La, Ce, Nd, Eu, U, Pu, etc.)
        self.element_list = element_list or [
            "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
            "Ho", "Er", "Tm", "Yb", "Lu", "U", "Pu", "Am"
        ]
        self.element_map = {sym: atomic_num for sym, atomic_num in np.core.defchararray.zfill(self.element_list, 3).items()}
        
        # Setup logger
        from algochem.utils.logger import setup_logger
        self.logger = setup_logger("PaperAlgorithm", verbose=verbose)
        self.logger.info(f"Initialized PaperAlgorithm for elements: {', '.join(self.element_list)}")
        
        # Initialize model/parameters (replace with paper's initialization logic)
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize trainable parameters or model components (paper-specific).
        
        Example: For an ML potential, this initializes a neural network; for a QM method,
        this sets basis set parameters or fitting coefficients.
        """
        # PLACEHOLDER: Replace with your algorithm's initialization
        self.model_coeffs = np.load(self.params.pretrained_coeffs_path) if self.params.use_pretrained else np.random.randn(50)
        self.logger.info(f"Model initialized with {len(self.model_coeffs)} coefficients.")

    def _validate_system(self, positions: np.ndarray, numbers: np.ndarray) -> None:
        """Validate input system (dimensions, element support)."""
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Positions must be (N, 3) array; got {positions.shape}")
        if numbers.ndim != 1 or len(numbers) != len(positions):
            raise ValueError(f"Numbers must be 1D array with length {len(positions)}; got {numbers.shape}")
        
        # Check for unsupported elements
        atomic_symbols = np.core.defchararray.ljust(numbers, 2)
        unsupported = [sym for sym in atomic_symbols if sym not in self.element_map]
        if unsupported:
            raise NotImplementedError(f"Unsupported elements: {', '.join(unsupported)} (Ln/An focus only)")

    def compute_energy(
        self,
        positions: np.ndarray,
        numbers: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[Tuple[bool, bool, bool]] = (False, False, False)
    ) -> float:
        """
        Compute the total potential energy of the system (paper's Equation X).
        
        Args:
            positions: Atomic coordinates in Angstroms (shape: (N, 3)).
            numbers: Atomic numbers (shape: (N,)).
            cell: Unit cell vectors for periodic systems (shape: (3, 3); optional).
            pbc: Periodic boundary conditions (optional).
        
        Returns:
            Total energy in eV (consistent with ASE/PySCF units).
        
        Raises:
            ValueError: If input dimensions are invalid.
            NotImplementedError: If unsupported elements are present.
        """
        self._validate_system(positions, numbers)
        self.logger.debug(f"Computing energy for {len(positions)}-atom system.")
        
        # PLACEHOLDER: Replace with PAPER'S ENERGY CALCULATION LOGIC
        # Example: Pairwise potential (delete this and insert your algorithm)
        dists = compute_pairwise_distances(positions, cell, pbc)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        atomic_symbols = np.core.defchararray.ljust(numbers, 2)
        ln_mask = np.isin(atomic_symbols, ["La", "Ce", "Nd"])
        
        # Apply paper's scaling for lanthanide interactions
        ln_scaling = self.params.ln_pair_scaling
        energy = -ln_scaling * np.sum(1.0 / dists[ln_mask][:, ln_mask])  # Ln-Ln pairs
        energy += np.sum(1.0 / dists[~ln_mask][:, ~ln_mask])  # Non-Ln pairs
        energy *= self.params.global_scaling_factor
        
        return float(energy)

    def compute_forces(
        self,
        positions: np.ndarray,
        numbers: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: Optional[Tuple[bool, bool, bool]] = (False, False, False)
    ) -> np.ndarray:
        """
        Compute atomic forces (negative gradient of energy; paper's Equation Y).
        
        Args:
            positions: Atomic coordinates in Angstroms (shape: (N, 3)).
            numbers: Atomic numbers (shape: (N,)).
            cell: Unit cell vectors (shape: (3, 3); optional).
            pbc: Periodic boundary conditions (optional).
        
        Returns:
            Atomic forces in eV/Angstrom (shape: (N, 3)).
        """
        # Use numerical gradients (replace with ANALYTICAL GRADIENTS from paper for speed)
        self.logger.debug("Computing forces via numerical gradients (replace with analytical for production).")
        forces = numerical_gradients(
            func=lambda pos: self.compute_energy(pos, numbers, cell, pbc),
            x=positions,
            h=self.params.numerical_gradient_h
        )
        return forces  # Numerical gradients return -dE/dr (correct force direction)

    def run_full_calculation(
        self,
        system: Dict[str, np.ndarray],
        properties: List[str] = ["energy", "forces"]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Unified interface to compute requested properties for a system.
        
        Args:
            system: Dictionary containing:
                - "positions": (N, 3) array of coordinates
                - "numbers": (N,) array of atomic numbers
                - "cell": (3, 3) array (optional, for periodic systems)
                - "pbc": (3,) boolean array (optional)
            properties: List of properties to compute ("energy", "forces", "stress").
        
        Returns:
            Dictionary of computed properties with consistent units.
        """
        results = {}
        cell = system.get("cell")
        pbc = system.get("pbc", (False, False, False))
        
        if "energy" in properties:
            results["energy"] = self.compute_energy(
                positions=system["positions"],
                numbers=system["numbers"],
                cell=cell,
                pbc=pbc
            )
        
        if "forces" in properties:
            results["forces"] = self.compute_forces(
                positions=system["positions"],
                numbers=system["numbers"],
                cell=cell,
                pbc=pbc
            )
        
        # Add "stress" (paper-specific) if needed for periodic systems
        if "stress" in properties and any(pbc):
            raise NotImplementedError("Stress calculation not yet implemented (add paper's logic here).")
        
        self.logger.info(f"Completed calculation: {', '.join(results.keys())} computed successfully.")
        return results
```

---

## 2. Hyperparameter Management (`algochem/core/params.py`)
Centralize all hyperparameters/parameters from the paper for reproducibility. Use `pydantic` for type checking and validation.

```python
"""
Hyperparameter and parameter management for the paper's algorithm.

Uses Pydantic to enforce type safety and ensure consistency with paper values.
"""
from pydantic import BaseModel, Field
from typing import Optional, Tuple

class AlgorithmParams(BaseModel):
    """
    Hyperparameters and parameters as defined in the paper.
    
    All default values should match the "default" or "best" settings reported
    in the paper's supplementary information.
    """
    # Global scaling (paper's Section 3.2)
    global_scaling_factor: float = Field(1.0, description="Global energy scaling factor (Eq. 4)")
    
    # Lanthanide-specific parameters (paper's Section 4.1)
    ln_pair_scaling: float = Field(1.2, description="Scaling for Ln-Ln pairwise interactions")
    
    # Numerical settings
    numerical_gradient_h: float = Field(1e-5, description="Step size for numerical gradients (eV/Å)")
    
    # ML/model settings (if applicable)
    use_pretrained: bool = Field(False, description="Use pre-trained coefficients from paper")
    pretrained_coeffs_path: str = Field("pretrained_coeffs.npy", description="Path to pre-trained coefficients")
    
    # Convergence criteria (for iterative algorithms)
    convergence_tol: float = Field(1e-6, description="Energy convergence tolerance (eV)")
    max_iterations: int = Field(1000, description="Maximum number of iterations")
```

---

## 3. ASE Calculator Integration (`algochem/calculators/ase_calc.py`)
Seamlessly integrate with the Atomic Simulation Environment (ASE) for structure optimization, molecular dynamics, and vibrational analysis—critical for reproducing paper benchmarks.

```python
"""
ASE (Atomic Simulation Environment) Calculator for the paper's algorithm.

This allows the algorithm to be used with ASE's full suite of tools:
- Structure optimization (BFGS, FIRE)
- Molecular dynamics (NVT, NPT)
- Phonon calculations
- Database management

Reference: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html
"""
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from typing import Dict, List, Union
from algochem.core.algorithm import PaperAlgorithm
from algochem.core.params import AlgorithmParams

class PaperAlgorithmASE(Calculator):
    """
    ASE-compatible calculator for the paper's algorithm.
    
    Implements standard ASE properties: energy (free_energy) and forces.
    Extensible to stress for periodic systems.
    """
    # Define supported properties (match ASE's naming convention)
    implemented_properties = ["energy", "forces"]
    default_parameters = {
        "global_scaling_factor": 1.0,
        "ln_pair_scaling": 1.2,
        "use_pretrained": False
    }

    def __init__(
        self,
        algorithm_params: Union[Dict, AlgorithmParams] = None,
        **kwargs
    ):
        """
        Initialize the ASE calculator.
        
        Args:
            algorithm_params: Hyperparameters for PaperAlgorithm (Dict or AlgorithmParams).
            **kwargs: Additional ASE calculator arguments (e.g., label).
        """
        # Initialize parent ASE Calculator class
        Calculator.__init__(self, **kwargs)
        
        # Merge default parameters with user input
        self.parameters.update(kwargs)
        
        # Initialize the core algorithm
        self.algorithm = PaperAlgorithm(
            params=algorithm_params or self.parameters,
            verbose=self.parameters.get("verbose", False)
        )

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes
    ) -> None:
        """
        Core calculation method (ASE required).
        
        Updates self.results with computed properties (energy, forces).
        
        Args:
            atoms: ASE Atoms object (system to compute).
            properties: List of properties to calculate (defaults to ["energy"]).
            system_changes: List of changes since last calculation (ASE internal).
        """
        # Standard ASE boilerplate: update atoms and reset results
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = {}

        # Prepare input for core algorithm
        system = {
            "positions": self.atoms.get_positions(),
            "numbers": self.atoms.get_atomic_numbers(),
            "cell": self.atoms.get_cell()[:],
            "pbc": self.atoms.get_pbc()
        }

        # Run core calculation
        computed_results = self.algorithm.run_full_calculation(
            system=system,
            properties=properties
        )

        # Map results to ASE's expected format
        if "energy" in computed_results:
            self.results["energy"] = computed_results["energy"]  # ASE uses eV
            self.results["free_energy"] = computed_results["energy"]  # Alias for ASE optimizers
        
        if "forces" in computed_results:
            self.results["forces"] = computed_results["forces"]  # ASE uses eV/Å
```

---

## 4. Utility Functions (`algochem/utils/math_utils.py`)
Reusable numerical utilities—critical for maintaining clean core code.

```python
"""
Mathematical utilities for computational chemistry calculations.
"""
import numpy as np
from typing import Callable, Tuple, Optional

def compute_pairwise_distances(
    positions: np.ndarray,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[Tuple[bool, bool, bool]] = (False, False, False)
) -> np.ndarray:
    """
    Compute pairwise distances between atoms (supports PBC via minimum image convention).
    
    Args:
        positions: (N, 3) array of atomic coordinates.
        cell: (3, 3) unit cell vectors (optional).
        pbc: (3,) boolean array for periodic boundary conditions (optional).
    
    Returns:
        (N, N) array of pairwise distances.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    
    # Apply minimum image convention for PBC
    if any(pbc) and cell is not None:
        inv_cell = np.linalg.inv(cell)
        diff_frac = np.dot(diff, inv_cell)
        diff_frac -= np.round(diff_frac) * np.array(pbc)[None, None, :]
        diff = np.dot(diff_frac, cell)
    
    return np.linalg.norm(diff, axis=-1)

def numerical_gradients(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradients using central difference formula.
    
    Args:
        func: Function to differentiate (takes x and returns a scalar).
        x: Input array (shape: (N, 3)) to compute gradients at.
        h: Step size for finite difference.
    
    Returns:
        Gradient array (shape: (N, 3)): dfunc/dx.
    """
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Central difference: (f(x+h) - f(x-h)) / (2h)
            x_plus = x.copy()
            x_plus[i, j] += h
            f_plus = func(x_plus)
            
            x_minus = x.copy()
            x_minus[i, j] -= h
            f_minus = func(x_minus)
            
            grad[i, j] = (f_plus - f_minus) / (2 * h)
    return grad
```

---

## 5. Example Script: Reproduce Paper Benchmark (`examples/run_optimization.py`)
This script replicates a key result from the paper (e.g., structure optimization of a La₂O₃ cluster) using the ASE calculator—**critical for demonstrating reproducibility**.

```python
#!/usr/bin/env python3
"""
Reproduce the structure optimization benchmark from the paper (Section 4.2, Figure 3).

System: La₂O₃ cluster (gas-phase, non-periodic)
Optimizer: BFGS (matching paper's settings)
Convergence: fmax = 0.01 eV/Å

Usage:
    python run_optimization.py
"""
from ase import Atoms
from ase.optimize import BFGS
from ase.io import write
from algochem.calculators.ase_calc import PaperAlgorithmASE

# --------------------------
# 1. Define System (Paper's Initial Structure)
# --------------------------
# Replace with the exact initial coordinates from the paper's SI
la2o3 = Atoms(
    symbols="La2O3",
    positions=[
        [0.0000, 0.0000, 0.0000],
        [3.1200, 0.0000, 0.0000],
        [1.5600, 1.0800, 0.0000],
        [1.5600, -1.0800, 0.0000],
        [1.5600, 0.0000, 1.6200]
    ],
    cell=[15.0, 15.0, 15.0],  # Large simulation box (gas-phase)
    pbc=(False, False, False)
)

# --------------------------
# 2. Initialize Calculator (Paper's Parameters)
# --------------------------
calc = PaperAlgorithmASE(
    algorithm_params={
        "global_scaling_factor": 1.0,
        "ln_pair_scaling": 1.2,
        "use_pretrained": True,  # Use paper's pre-trained coefficients
        "pretrained_coeffs_path": "paper_systems/pretrained_la2o3_coeffs.npy"
    },
    verbose=True
)
la2o3.calc = calc

# --------------------------
# 3. Run Optimization (Paper's Settings)
# --------------------------
print(f"Initial Energy: {la2o3.get_potential_energy():.4f} eV")
print(f"Initial Max Force: {np.max(np.linalg.norm(la2o3.get_forces(), axis=1)):.4f} eV/Å")

# Initialize BFGS optimizer (matching paper's optimizer)
opt = BFGS(
    atoms=la2o3,
    trajectory="la2o3_optimization.traj",
    logfile="la2o3_optimization.log",
    maxstep=0.2  # Paper's max step size
)

# Run until convergence (paper's fmax)
print("\nStarting structure optimization...")
opt.run(fmax=0.01)

# --------------------------
# 4. Save and Print Results
# --------------------------
write("la2o3_optimized.xyz", la2o3)
print(f"\nOptimized Energy: {la2o3.get_potential_energy():.4f} eV")
print(f"Optimized Max Force: {np.max(np.linalg.norm(la2o3.get_forces(), axis=1)):.4f} eV/Å")
print(f"Optimized Structure saved to: la2o3_optimized.xyz")

# Compare to paper's results (add your paper's reference values here)
paper_optimized_energy = -12.34  # Replace with paper's value
print(f"\nPaper Reference Energy: {paper_optimized_energy:.4f} eV")
print(f"Absolute Difference: {abs(la2o3.get_potential_energy() - paper_optimized_energy):.4f} eV")
```

---

## 6. Packaging & Installation (`pyproject.toml`)
Modern Python packaging for easy installation and dependency management (compatible with `pip`).

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "NLFMP"
version = "0.1.0"

description = "Python implementation of the algorithm proposed in [Paper Title]"
long_description = file: "README.md"
long_description_content_type = "text/markdown"
license = { file = "LICENSE" }
keywords = ["computational chemistry", "lanthanides", "machine learning potential", "quantum chemistry"]
classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Physics",
    "Intended Audience :: Science/Research"
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "ase>=3.22.1",
    "pyscf>=2.3.0",  # If using PySCF integration
    "pydantic>=2.0.0",  # For parameter validation
    "matplotlib>=3.5.0"  # For plotting (examples)
]

```

---

To fully implement the paper’s algorithm, follow these steps:
1. **Update `core/algorithm.py`**:
   - Replace the placeholder `_initialize_model` with the paper’s model/parameter initialization.
   - Rewrite `compute_energy` using the **exact equations** from the paper (e.g., neural network forward pass, quantum chemical integral evaluation, or force field terms).
   - Replace the numerical `compute_forces` with **analytical gradients** (critical for speed in ML potentials/DFT methods).
2. **Update `core/params.py`**:
   - Add all hyperparameters/parameters from the paper (e.g., neural network layers, basis set sizes, fitting parameters).
   - Set default values to match the paper’s "best" or "default" settings.
3. **Add PySCF Integration**:
   - If the algorithm relies on PySCF (e.g., for SCF calculations), implement `calculators/pyscf_interface.py` to wrap the algorithm into PySCF’s `Mole` class.
4. **Reproduce Paper Examples**:
   - Add input files (XYZ/CIF) from the paper to `examples/paper_systems/`.
   - Write additional example scripts to reproduce **all key benchmarks** (energy/force errors, convergence curves, ablation studies).
5. **Write Unit Tests**:
   - Add tests in `tests/` to verify that the algorithm produces **exact values** for small systems (e.g., a single La atom) as reported in the paper.

---

## 8. Installation & Usage
1. **Install the package in development mode**:
   ```bash
   cd algochem
   pip install -e .
   ```
2. **Run the example script**:
   ```bash
   cd examples
   python run_optimization.py
   ```
