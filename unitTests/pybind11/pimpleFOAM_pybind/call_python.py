# %%
from pimpleFOAM_pybind import pimpleFOAM_pybind
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

def solve_all_times(a):
    '''
    Solve all time steps using a single function in C++.
    '''
    a.solveAll()

def solve_all_python(a):
    '''
    Create the loop in python, calling a C++ function to solve each time step.
    '''
    # extract times
    deltaT = a.getDeltaT()
    startT = a.getStartTime()
    finalT = a.getEndTime()
    finalT = 0.004
    # define times in python
    times = np.arange(startT, finalT, step=deltaT) + deltaT
    for t in times:
        a.solveOneTimeStep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pimpleFoam with UnsteadyNSTurb")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-solveAllC", action="store_true")
    group.add_argument("-solveAllpython", action="store_true")
    args = parser.parse_args()

    # Instantiate OF object
    a = pimpleFOAM_pybind(["."])


    if args.solveAllC:
        solve_all_times(a)

    if args.solveAllpython:
        solve_all_python(a)
