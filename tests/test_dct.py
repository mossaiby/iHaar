import numpy as np, sys; sys.path.insert(0,'.')
from ihaar.phasefield import SAVPhaseField
rng=np.random.default_rng(3); u0 = 0.1+0.05*rng.standard_normal((96,96))
for order in (2,4):
    a = SAVPhaseField(96,2,0.02,1e-5,'CH',order); b = SAVPhaseField(96,2,0.02,1e-5,'CH',order,transform='dct')
    a.initialize(u0); b.initialize(u0)
    for i in range(50): a.step(); b.step()
    print(order, 'gemm vs dct', np.abs(a.u-b.u).max())
# 3D
s = SAVPhaseField(24,3,0.05,1e-4,'CH',4); s.initialize(0.1+0.05*rng.standard_normal((24,24,24)))
d = SAVPhaseField(24,3,0.05,1e-4,'CH',4,transform='dct'); d.initialize(s.u.copy())
m0=s.mass(s.u)
for i in range(30): s.step(); d.step()
print('3D gemm vs dct', np.abs(s.u-d.u).max(), 'mass', s.mass(s.u)-m0)
