import numpy as np, sys, time; sys.path.insert(0,'.')
from ihaar.elasticity import SmoothField, WilliamsField, ElasticityTensor
from ihaar.elasticity_bc import ElasticityMixed
f = SmoothField()
# (1) all Dirichlet must reproduce the original solver
t = np.linspace(0,1,17); a = ElasticityTensor(t,t); U,V,_,_ = a.solve(f, body=f.body)
b = ElasticityMixed(t,t,{}); Ue,Ve,its,_ = b.solve(f, body_fn=f.body)
print('all-D agreement', np.abs(Ue[1:-1,1:-1]-U).max(), np.abs(Ve[1:-1,1:-1]-V).max(), 'its', its)
for bc in ({'T':'T'}, {'T':'T','R':'T'}, {'L':'T','B':'T'}, {'T':'T','B':'T'}):
    prev=None; print('bc', bc)
    for N in [8,16,32,64,128]:
        t=np.linspace(0,1,N+1); el=ElasticityMixed(t,t,bc); T=time.time()
        Ue,Ve,its,_=el.solve(f, body_fn=f.body)
        e=el.errors_gauss(Ue,Ve,f); ee=el.edge_error(Ue,Ve,f)
        print(f'  N={N:4d} H1 {e:.3e} rate {"" if prev is None else f"{np.log2(prev/e):.2f}"} edge {ee:.2e} its {its} condV {el.condV:.1f} {time.time()-T:.2f}s'); prev=e
