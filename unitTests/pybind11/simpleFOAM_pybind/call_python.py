#%%
from simpleFOAM_pybind import simpleFOAM_pybind
from scipy.sparse.linalg import spsolve
import numpy as np

#Instantiate OF object
a = simpleFOAM_pybind(["."])
u = a.getU()
p = a.getP()
u2 = u.copy()
resU = np.zeros([u.size])
resP = np.zeros([p.size])
phi = a.getPhi()
u2[3] = 1.5
p[0] = 1.5
phi[0] = 1.5
a.setU(u2)
a.printU()

def residual(x):
    u_set = x[0:int(len(x)/4*3)]
    p_set = x[int(len(x)/4*3)::]
    print(len(u_set))
    print(len(p_set))
    a.setU(u_set)
    a.setP(p_set)
    res = a.getResidual()
    return res


up = np.ma.concatenate((u,p),axis=0)

print(np.linalg.norm(residual(up)))
up[0] = 100
print(np.linalg.norm(residual(up)))
exit()

#%%
# a.printU()
# a.printP()
# a.printPhi()
u[0] = 1.5
res = a.getResidual()
print(np.linalg.norm(res))
u[0] = 3
res = a.getResidual()
print(np.linalg.norm(res))


# %%
