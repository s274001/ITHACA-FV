# %%
from pimpleFOAM_pybind import pimpleFOAM_pybind
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Instantiate OF object
a = pimpleFOAM_pybind(["."])
#U = a.getU()
#print(U)
#a.printU()
a.solveAll()
