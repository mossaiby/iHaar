import numpy as np, sys, time; sys.path.insert(0,'.')
from ihaar.elasticity import WilliamsField, dorfler, bisect
from ihaar.elasticity_bc import ElasticityMixed
fld = WilliamsField(apex=(0.0, 1.0))      # physical [0,1]x[-1,0] shifted: top edge = traction-free face
x = np.linspace(0.05,1,5); a,b,c,d = fld.grad(x, 1+0*x); G=fld.G; L=fld.lmb
print('exact traction on top edge:', np.abs(G*(b+c)).max(), np.abs(L*a+(L+2*G)*d).max())
bc = {'T':'T'}
for N in [8,16,32,64,128]:
    t=np.linspace(0,1,N+1); el=ElasticityMixed(t,t,bc); Ue,Ve,its,_=el.solve(fld)
    print('uniform', 2*N*N, f'{el.errors_gauss(Ue,Ve,fld):.3e}', its)
tx=ty=np.linspace(0,1,9)
for cyc in range(40):
    el=ElasticityMixed(tx,ty,bc); T=time.time(); Ue,Ve,its,_=el.solve(fld)
    e=el.errors_gauss(Ue,Ve,fld); etx,ety=el.indicators(Ue,Ve); dofs=2*el.nunk
    if cyc%3==0: print(cyc, dofs, f'{e:.3e} its {its} condV {el.condV:.1f} hmin {min(np.diff(tx).min(),np.diff(ty).min()):.1e} {time.time()-T:.1f}s')
    if dofs>4e4: print(cyc, dofs, f'{e:.3e} its {its}'); break
    mx,my=dorfler(etx,ety,0.6); tx=bisect(tx,mx); ty=bisect(ty,my)
