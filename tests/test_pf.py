import numpy as np, sys, time
sys.path.insert(0,'.')
from ihaar.phasefield import SAVPhaseField
rng = np.random.default_rng(1)
for model, dt in [('CH',1e-5), ('AC', 1e-1)]:
    s = SAVPhaseField(128, 2, eps=0.01, dt=dt, model=model, order=4)
    s.initialize(0.1 + 0.05*rng.standard_normal((128,128)) if model=='CH' else 0.9*rng.uniform(-1,1,(128,128)))
    m0 = s.mass(s.u); Em=[]; E=[]
    t=time.time()
    for n in range(400):
        s.step(); Em.append(s.modified_energy()); E.append(s.energy(s.u, s.uhat))
    dEm = np.diff(Em[1:]); print("first dEm", np.diff(Em)[:3]); dE=np.diff(E)
    print(model, 'mass drift', abs(s.mass(s.u)-m0), 'max dEmod', dEm.max(), 'max dE', dE.max(),
          'E0->E', E[0], E[-1], 'xi=r/sqrt', s.r/np.sqrt(s.E1(s.u)+1), f'{(time.time()-t)/400*1e3:.1f} ms/step')

# stabilized SAV: modified energy (with S/2||u^{n+1}-u^n||^2) must decrease
s = SAVPhaseField(64, 2, eps=0.01, dt=1e-4, model='CH', order=4, S=4.0)
s.initialize(0.1 + 0.05*np.random.default_rng(2).standard_normal((64,64)))
Em=[]
for n in range(300):
    s.step()
    if n >= 1: Em.append(s.modified_energy())
print('stabilized SAV: max dEmod', np.diff(Em).max(), '(must be <= 0)')
