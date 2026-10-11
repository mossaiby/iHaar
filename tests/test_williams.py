import numpy as np, sys
sys.path.insert(0,'.')
from ihaar.elasticity import WilliamsField, LAM_I, OMEGA
w = WilliamsField()
print('char eq residual', np.sin(LAM_I*OMEGA)+LAM_I*np.sin(OMEGA))
# gradient check
x, y, e = 0.37, 0.52, 1e-6
gx = [(np.array(w.disp(x+e,y))-np.array(w.disp(x-e,y)))/(2*e), (np.array(w.disp(x,y+e))-np.array(w.disp(x,y-e)))/(2*e)]
g = w.grad(np.array(x), np.array(y)); print('grad err', np.abs(np.array([gx[0][0],gx[1][0],gx[0][1],gx[1][1]]) - np.array(g)).max())
# Navier residual via FD of the analytic gradient
def stress(x,y):
    a,b,c,d = w.grad(np.array(x),np.array(y)); div=a+d
    return w.lmb*div+2*w.G*a, w.G*(b+c), w.lmb*div+2*w.G*d
e=1e-5
sxx1,sxy1,_=stress(x+e,y); sxx0,sxy0,_=stress(x-e,y)
_,sxy3,syy3=stress(x,y+e); _,sxy2,syy2=stress(x,y-e)
print('Navier residual', (sxx1-sxx0)/(2*e)+(sxy3-sxy2)/(2*e), (sxy1-sxy0)/(2*e)+(syy3-syy2)/(2*e))
# traction on faces: face at global angle bis+-3pi/4 from apex (1,1)
for sgn in (+1,-1):
    ang = w.bis + sgn*OMEGA/2; rr=0.3
    px, py = 1+rr*np.cos(ang), 1+rr*np.sin(ang)
    sxx,sxy,syy = stress(px,py); n = np.array([-np.sin(ang), np.cos(ang)])
    print('traction face', sgn, sxx*n[0]+sxy*n[1], sxy*n[0]+syy*n[1])
