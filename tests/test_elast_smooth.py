import numpy as np, sys; sys.path.insert(0,'.')
from ihaar.elasticity import SmoothField, ElasticityTensor
for nu in (0.3,):
    f = SmoothField(nu=nu); prev=None
    for N in [8,16,32,64,128]:
        t=np.linspace(0,1,N+1); el=ElasticityTensor(t,t,nu=nu); U,V,its,_=el.solve(f,body=f.body)
        e,ei=el.errors(U,V,f); print(N, e, ei, its, '' if prev is None else np.log2(prev/e)); prev=e
