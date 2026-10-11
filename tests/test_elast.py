import numpy as np, sys, time
sys.path.insert(0,'.')
from ihaar.elasticity import WilliamsField, ElasticityTensor, dorfler, bisect
fld = WilliamsField()
print('uniform')
for N in [8,16,32,64,128]:
    t = np.linspace(0,1,N+1); T=time.time()
    el = ElasticityTensor(t,t); U,V,its,h = el.solve(fld, tol=1e-10)
    eH1, einf = el.errors(U,V,fld)
    print(N, 2*N*N, f'H1 {eH1:.3e} inf {einf:.3e} its {its} t {time.time()-T:.2f}s')
print('adaptive')
tx = ty = np.linspace(0,1,9)
for cyc in range(14):
    el = ElasticityTensor(tx,ty); T=time.time(); U,V,its,h = el.solve(fld, tol=1e-10)
    eH1, einf = el.errors(U,V,fld); ex,ey = el.indicators(U,V)
    eta = np.sqrt(ex.sum()+ey.sum())
    print(cyc, el.shape, 2*el.shape[0]*el.shape[1], f'H1 {eH1:.3e} inf {einf:.3e} eta {eta:.3e} its {its} hmin {min(np.diff(tx).min(),np.diff(ty).min()):.1e} condV {el.condV:.1e} t {time.time()-T:.2f}')
    if 2*el.shape[0]*el.shape[1] > 6e4: break
    mx,my = dorfler(ex,ey,0.5); tx = bisect(tx,mx); ty = bisect(ty,my)
