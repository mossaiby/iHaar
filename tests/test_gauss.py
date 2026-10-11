import numpy as np, sys; sys.path.insert(0,'.')
from ihaar.elasticity import SmoothField, WilliamsField, ElasticityTensor
from ihaar.core import Grid1D
# 1D reconstruction check: spline from (U,g) must reproduce quadratics exactly
g = Grid1D(np.sort(np.r_[0,1,np.random.default_rng(1).uniform(0,1,9)]),'D')
p = lambda x: 3*x**2-2*x+1; xq=np.linspace(0,1,101); E,Ed=g.eval_matrices(xq)
w = np.r_[p(0.), p(g.x), p(1.)]; print('quad repro', np.abs(E@w-p(xq)).max(), np.abs(Ed@w-(6*xq-2)).max())
for fld, body in ((SmoothField(), SmoothField().body), (WilliamsField(), None)):
    for N in (16,32,64):
        t=np.linspace(0,1,N+1); el=ElasticityTensor(t,t); U,V,_,_=el.solve(fld,body=body)
        print(type(fld).__name__, N, 'mid', el.errors(U,V,fld)[0], 'gauss', el.errors_gauss(U,V,fld))
