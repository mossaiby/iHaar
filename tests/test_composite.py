"""Local refinement by nested composite patches: the coupled iteration converges, matches the accuracy of
the tensor grid with the same local resolution, and needs far fewer unknowns."""
import sys, numpy as np
sys.path.insert(0, '.')
from ihaar.composite import CompositePoisson, poisson_tensor, corner_knots, h1_error

def u(x, y):                                # harmonic, gradient singular at the origin
    r, th = np.hypot(x, y), np.arctan2(y, x); return r ** (2 / 3) * np.sin(2 * th / 3)
def grad(x, y):
    r = np.hypot(x, y) + 1e-300; th = np.arctan2(y, x)
    ur, ut = (2 / 3) * r ** (-1 / 3) * np.sin(2 * th / 3), (2 / 3) * r ** (-1 / 3) * np.cos(2 * th / 3)
    return ur * np.cos(th) - ut * np.sin(th), ur * np.sin(th) + ut * np.cos(th)

n, L = 16, 6
cp = CompositePoisson(n, L, u); hist = cp.solve()
ec = cp.h1_error(grad)
t = corner_knots(n, L); et = h1_error([poisson_tensor(t, t, u)], grad)
print("sweeps", len(hist), "last update %.1e" % hist[-1])
print("composite: dofs", cp.dofs, "eH1 %.3e | tensor: dofs" % ec, (t.size - 1) ** 2, "eH1 %.3e" % et)
assert hist[-1] < 1e-12 and len(hist) < 20
assert abs(ec - et) < 0.05 * et
assert cp.dofs < 0.5 * (t.size - 1) ** 2
# the composite solution solves the same problem from any starting guess (unique fixed point)
cp2 = CompositePoisson(n, L, u)
for p in cp2.P: p.U = np.random.default_rng(0).standard_normal(p.U.shape)
cp2.solve()
print("difference from random start: %.1e" % max(np.abs(a.U - b.U).max() for a, b in zip(cp.P, cp2.P)))
assert max(np.abs(a.U - b.U).max() for a, b in zip(cp.P, cp2.P)) < 1e-10
print("OK")
