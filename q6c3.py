#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg, integrate
from scipy.sparse.linalg import LinearOperator, eigsh
from matplotlib import pyplot as plt
from matplotlib import rc
import minitn.bases.dvr as dvr
import minitn
rc('font', family='Times New Roman')
rc('text', usetex=True)

DTYPE = np.complex128


INF = 1.0e14
dim = 200
hbar = 1.0
omega = 1.0
m = 1.0

def v_rst(tup, y=None):
    if y is None:
        x, y = tup
    else:
        x = tup
    if - 0.5 <= x <= 0.5 and - 0.5 <= y <= 0.5:
        v = 0.0
    elif (x - 0.5)** 2 + y ** 2 <= 0.25:
        v = 0.0
    elif (x + 0.5)** 2 + y ** 2 <= 0.25:
        v = 0.0
    else:
        v = INF
    return v

def vfunc(x):
    return np.zeros_like(x)

def main():
    solver = dvr.PO_DVR([(-1, 1, dim), (-0.5, 0.5, dim)], hbar=hbar, m_e=m, fast=False)
    solver.set_v_func([vfunc, vfunc], v_rst=v_rst) 
    try:
        e = np.loadtxt('e3.txt')
        v = np.loadtxt('v3.txt')
    except:
        solver = dvr.PO_DVR([(-1, 1, dim), (-0.5, 0.5, dim)], hbar=hbar, m_e=m, fast=False)
        solver.set_v_func([vfunc, vfunc], v_rst=v_rst) 
        e, v = solver.solve(n_state=dim)
        np.savetxt('e3.txt', e)
        np.savetxt('v3.txt', v)

    x, y = solver.grid_points_list[:2]
    shape = solver.n_list[:2]
    x, y = np.meshgrid(x, y)
    cap = np.zeros_like(x)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cap[i, j] = v_rst(x[i, j], y[i, j])

    for i in [0, 1, 2, 5, 10, 20, 50, 100, 150, 195, 198, 199]:
        vec = v[i]
        print('E_{}: {}'.format(i, e[i]))
        z_lim = (np.max(np.abs(vec)) * 15) / 10
        bound = np.linspace(-z_lim, z_lim, 100)
        vec = np.reshape(vec, shape)        
        plt.contourf(x, y, vec, bound, cmap='seismic')
        plt.colorbar()
        plt.xlim(-1., 1.)
        plt.ylim(-1, 1)
        plt.savefig('q6c-{}.pdf'.format(i))
        plt.close()


if __name__ == '__main__':
    main()
