"""ihaar: integrated-Haar wavelet collocation with fast tensor solvers (CPU/GPU)."""
from .core import Grid1D, corrected_D2, Eig1D, TensorFDM, NeumannDCT
from .phasefield import SAVPhaseField
from .elasticity import WilliamsField, SmoothField, ElasticityTensor
