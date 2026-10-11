"""Theorem 2: D2 = A^{-1} L has real, simple, nonpositive eigenvalues on every grid."""
import sys, numpy as np
sys.path.insert(0, '.')
from ihaar.core import Grid1D
rng = np.random.default_rng(0); worst = 0.0; pos = 0
for trial in range(500):
    N = int(rng.integers(4, 80)); lr = rng.uniform(-np.log(3), np.log(3), N - 1)
    h = np.exp(np.clip(np.r_[0, np.cumsum(lr)], -6, 6)); t = np.r_[0, np.cumsum(h)] / h.sum()
    for bc in "DN":
        lam = np.linalg.eigvals(Grid1D(t, bc).D2)
        worst = max(worst, np.abs(lam.imag).max() / np.abs(lam).max())
        lr_ = lam.real if bc == "D" else np.sort(lam.real)[:-1]
        pos += int((lr_ >= 0).any())
print("max relative imaginary part:", worst, "| grids with a nonnegative (nonkernel) eigenvalue:", pos)
assert worst < 1e-10 and pos == 0
print("OK")
