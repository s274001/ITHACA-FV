from simpleFOAM_pybind import simpleFOAM_pybind
from scipy.sparse.linalg import spsolve

#Instantiate OF object
a = simpleFOAM_pybind(["."])
u = a.getU()
p = a.getP()
phi = a.getPhi()
u[0,0] = 1.5
p[0] = 1.5
phi[0] = 1.5
a.printU()
a.printP()
a.printPhi()

