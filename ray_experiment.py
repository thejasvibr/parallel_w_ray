# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:36:24 2022

@author: theja
"""

import ray 
import time
import numpy as np 
import cppyy
try:
    cppyy.add_include_path('../np_vs_eigen/eigen/')
    cppyy.include('../np_vs_eigen/example_eigen.cpp')
    MatrixXd = cppyy.gbl.Eigen.MatrixXd
    VectorXd = cppyy.gbl.Eigen.VectorXd
except ImportError:
    pass

posns = np.random.normal(0,2,15).reshape(-1,3)
d = np.random.choice(np.linspace(-0.5,0.5,50), posns.shape[0]-1)

ray.init()
@ray.remote 
def cpp_spiesberger_wahlberg(array_geom_np, d_np, **kwargs):
    '''

    '''
    mic0 = np.zeros((1,3))
    array_geomnp_copy = array_geom_np.copy()
    if np.sum(array_geom_np[0,:]) != 0:
        mic0 = array_geomnp_copy[0,:]
        array_geomnp_copy -= array_geomnp_copy[0,:]

    rows, cols = array_geom_np.shape
    array_geom = MatrixXd(rows, cols)
    for i in range(rows):
        for j in range(cols):
            array_geom[i,j] = array_geom_np[i,j]

    d = VectorXd(rows-1)
    for i, value in enumerate(d_np):
        d[i] = value

    solutions = cppyy.gbl.spiesberger_wahlberg_solution(array_geom, d, 343.0)
    s = list(map(lambda X: np.array(list(X)), solutions))
    
    return s
print('starting loop...')
start = time.perf_counter_ns()
all_results = []
for i in range(100):
    
    posns = np.random.normal(0,2,15).reshape(-1,3)
    d = np.random.choice(np.linspace(-0.5,0.5,50), posns.shape[0]-1)

    all_results.append(cpp_spiesberger_wahlberg.remote(posns, d))
results = ray.get(all_results)
print(f'{(time.perf_counter_ns()-start)/1e9} s')