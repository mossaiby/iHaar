import numpy as np, sys
sys.path.insert(0, '.')
from ihaar.core import Grid1D, integrated_haar_dirichlet, corrected_D2, haar_matrix

rng = np.random.default_rng(0)
# non-uniform knots
t = np.sort(np.concatenate([[0, 1], rng.uniform(0, 1, 23)]))
g = Grid1D(t, 'D')
x = g.x
Phi, Hv = integrated_haar_dirichlet(t, x)
assert np.allclose(Hv, np.eye(len(x)))
Pi = np.linalg.inv(Phi)
print('D2 compact vs integrated-Haar :', np.abs(g.D2 - Pi).max() / np.abs(Pi).max())
print('e0 :', np.abs(g.e0 + Pi @ (1 - x)).max(), ' e1:', np.abs(g.e1 + Pi @ x).max())
# D1 check
tl, tr = t[:-1], t[1:]; h = tr - tl; m = .5*(tl+tr)
X = x[:, None]
P1 = np.where(X < tl, 0, np.where(X < tr, X - tl, h))
P1 = P1 - h*(1-m)
D1ref = P1 @ Pi
print('D1 compact vs integrated-Haar :', np.abs(g.D1 - D1ref).max()/np.abs(D1ref).max())
f0ref = -(P1 @ Pi @ (1-x)) - 1; f1ref = -(P1 @ Pi @ x) + 1
print('f0,f1:', np.abs(g.f0-f0ref).max(), np.abs(g.f1-f1ref).max())
# Haar transform equivalence: Haar coefficients of u'' are orthogonal transform of c
N = 32; W = haar_matrix(N); print('Haar orthogonal:', np.abs(W@W.T - np.eye(N)).max())
# symmetry on uniform grids
for bc in 'DN':
    gu = Grid1D.uniform(64, bc=bc)
    print(bc, 'uniform D2 symmetric:', np.abs(gu.D2 - gu.D2.T).max()/np.abs(gu.D2).max(),
          ' eig range*h^2:', np.linalg.eigvalsh(gu.D2)[[0,-1]]*gu.h[0]**2)
# mass: h^T D2 = 0 for Neumann on non-uniform grid
gn = Grid1D(t, 'N'); print('Neumann h^T D2 =', np.abs(gn.h @ gn.D2).max())
# accuracy in 1D (Dirichlet, non-homog), order 2
for bc in 'DN':
    errs=[]; errs4=[]
    for N in [16, 32, 64, 128, 256]:
        gu = Grid1D.uniform(N, bc=bc); x = gu.x
        if bc == 'D':
            ue = lambda x: np.exp(np.sin(2*x)); 
            from numpy import exp, sin, cos
            upp = lambda x: exp(sin(2*x))*(4*cos(2*x)**2 - 4*sin(2*x))
            U = ue(x); c = gu.d2(U, ue(0.), ue(1.))
        else:
            ue = lambda x: np.cos(np.pi*x) + 0.3*np.cos(3*np.pi*x)
            upp = lambda x: -np.pi**2*np.cos(np.pi*x) - 0.3*9*np.pi**2*np.cos(3*np.pi*x)
            U = ue(x); c = gu.D2 @ U
            c4 = corrected_D2(gu) @ U; errs4.append(np.abs(c4 - upp(x)).max())
        errs.append(np.abs(c - upp(x)).max())
    print(bc, 'D2 truncation orders', np.round(np.log2(np.array(errs[:-1])/errs[1:]),2),
          ' corrected:', np.round(np.log2(np.array(errs4[:-1])/errs4[1:]),2) if errs4 else '')
