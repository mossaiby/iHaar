"""CPU vs GPU consistency check (requires CuPy and a CUDA device).
    python tests/test_gpu.py
"""
import sys, numpy as np
sys.path.insert(0, '.')
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
except Exception as e:
    print("CuPy / CUDA device not available:", e); sys.exit(1)
from ihaar.phasefield import SAVPhaseField
from ihaar.elasticity import SmoothField, ElasticityTensor
rng = np.random.default_rng(0)
ok = True
for tr in ("gemm", "dct"):
    for S in (0.0, 4.0):
        u0 = 0.1 + 0.05 * rng.standard_normal((128, 128))
        a = SAVPhaseField(128, 2, 0.01, 1e-5, "CH", 4, transform=tr, S=S); a.initialize(u0)
        try:
            b = SAVPhaseField(128, 2, 0.01, 1e-5, "CH", 4, transform=tr, S=S, xp=cp); b.initialize(cp.asarray(u0))
        except RuntimeError as e:
            print(f"[{tr}] skipped: {e}"); continue
        for _ in range(50): a.step(); b.step()
        d = float(np.abs(a.u - cp.asnumpy(b.u)).max()); ok &= d < 1e-10
        print(f"phase-field CH {tr:4s} S={S}: max |CPU-GPU| = {d:.2e}")
f = SmoothField(); t = np.linspace(0, 1, 33)
U, V, its, _ = ElasticityTensor(t, t).solve(f, body=f.body)
Ug, Vg, itg, _ = ElasticityTensor(t, t, xp=cp).solve(f, body=f.body)
d = float(max(np.abs(U - cp.asnumpy(Ug)).max(), np.abs(V - cp.asnumpy(Vg)).max())); ok &= d < 1e-8
print(f"elasticity (Dirichlet) : max |CPU-GPU| = {d:.2e}, GMRES its CPU {its} / GPU {itg}")
print("ALL OK" if ok else "MISMATCH"); sys.exit(0 if ok else 2)
