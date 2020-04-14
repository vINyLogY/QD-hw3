#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function
from functools import partial

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc

from grids import Coordinate, gaussian_wfn, Propagator

rc('font', family='Times New Roman')
rc('text', usetex=True)

op = Coordinate.mop
DTYPE = np.complex128

# General parameters
# basic units: fs, eV.
x1 = -8.67
x2 = 5.62
m = 86.65
k1 = 0.02
k2 = 0.02
hbar = 0.6582
omega0 = 0.44 / hbar
lambda0 = 0.19

# Laser
tw = 10.0
tc = 50.0
omega = 2.48 / hbar
e0_mu = 0.2

exp = np.exp
cos = np.cos
basis = Coordinate(-24.4, 35.6, 1024, hbar=hbar)


def v1(x): return 0.5 * k1 * (x - x1)**2


def v2(x): return 0.5 * k2 * (x - x2)**2 + hbar * omega0


def e_mu(t): return e0_mu * exp(-((t - tc) / tw)**2) * cos(omega * (t - tc))


def t_(p): return p**2 / (2.0 * m)


def t_func(p):
    t1 = t_(p)
    zero = np.zeros_like(t1)
    return np.array([[t1, zero], [zero, t1]], dtype=DTYPE)


def s12a(wfn):
    chi1, chi2 = basis(wfn)
    chi1 = chi1 / basis.norm(chi1)
    chi2 = chi2 / basis.norm(chi2)
    return np.abs(basis.inner_product(chi1, chi2))**2


def s12b(wfn):
    chi1, chi2 = basis(wfn)
    return np.abs(basis.inner_product(chi1, chi2))**2



def q3_2():
    chi = gaussian_wfn(x0=x1)
    def v12(x): return lambda0 * np.ones_like(x)

    def v_func(x):
        return np.array([[v1(x), v12(x)], [v12(x), v2(x)]], dtype=DTYPE)

    def init_wfn(x):
        return np.array([chi(x), chi(x)], dtype=DTYPE) / np.sqrt(2.0)

    propergator = Propagator(init_wfn, basis, v_func, t_func)
    dt = 0.1
    zipped = []
    for t, wfn in propergator(dt=dt, max_iter=1000, renormalize=True):
        potential = basis.inner_product(wfn, op(v_func)(wfn))
        wfn_p = basis.ft(wfn)
        kinetic = basis.conj.inner_product(wfn_p, op(t_func)(wfn_p))
        energy = kinetic + potential
        sa, sb = s12a(wfn), s12b(wfn)
        print('Time: {:.2f}; S_12(a): {:.8f}; S_12(b): {:.8f}; Energy: {:.8f}'.format(
            t, sa, sb, energy))
        zipped.append((t, sa, sb, energy))

        psi1, psi2 = basis(wfn)

        fig = plt.figure()
        ax2 = fig.add_axes([0.1, 0.52, 0.8, 0.4],
                           xticklabels=[], ylim=(0, 1))
        ax1 = fig.add_axes([0.1, 0.08, 0.8, 0.4],
                           ylim=(0, 1))

        ax1.plot(basis(), np.abs(psi1)**2, '-', label='1', c='b')
        ax1.set_ylabel('Probability Density')
        ax1.set_xlabel('Position')
        ax11 = ax1.twinx()
        ax11.plot(basis(), basis(v1), '--', c='grey')
        ax11.set_ylabel('Potential')
        ax11.set_ylim(0, 5)
        ax1.legend()

        ax2.plot(basis(), np.abs(psi2)**2, '-', label='2', c='y')
        ax2.set_ylabel('Probability Density')
        ax22 = ax2.twinx()
        ax22.plot(basis(), basis(v2), '--', c='grey')
        ax22.set_ylabel('Potential')
        ax22.set_ylim(0, 5)
        ax2.legend()

        plt.savefig('q3-2/wfn{:08.0f}.png'.format(t / dt), dpi=300)
        plt.close()
    np.savetxt('q3-2.txt', np.array(zipped), header='Time S_12(a) S_12(b) Energy')


def q3_3():
    chi = gaussian_wfn(x0=x1)
    def v12(x, t): return (lambda0 - e_mu(t)) * np.ones_like(x)

    def v_func(x, t=0.0):
        return np.array([[v1(x), v12(x, t)], [v12(x, t), v2(x)]], dtype=DTYPE)

    def init_wfn(x):
        return np.array([chi(x), np.zeros_like(x)], dtype=DTYPE)

    propergator = Propagator(init_wfn, basis, v_func, t_func)
    dt = 0.1
    zipped = []
    for t, wfn in propergator(dt=dt, max_iter=1000, renormalize=True):
        potential = basis.inner_product(wfn, op(v_func)(wfn))
        wfn_p = basis.ft(wfn)
        kinetic = basis.conj.inner_product(wfn_p, op(t_func)(wfn_p))
        energy = kinetic + potential
        sa, sb = s12a(wfn), s12b(wfn)
        print('Time: {:.2f}; S_12(a): {:.8f}; S_12(b): {:.8f}; Energy: {:.8f}'.format(
            t, sa, sb, energy))
        zipped.append((t, sa, sb, energy))

        psi1, psi2 = basis(wfn)

        fig = plt.figure()
        ax2 = fig.add_axes([0.1, 0.52, 0.8, 0.4],
                           xticklabels=[], ylim=(0, 1))
        ax1 = fig.add_axes([0.1, 0.08, 0.8, 0.4],
                           ylim=(0, 1))

        ax1.plot(basis(), np.abs(psi1)**2, '-', label='1', c='b')
        ax1.set_ylabel('Probability Density')
        ax1.set_xlabel('Position')
        ax11 = ax1.twinx()
        ax11.plot(basis(), basis(v1), '--', c='grey')
        ax11.set_ylabel('Potential')
        ax11.set_ylim(0, 5)
        ax1.legend()

        ax2.plot(basis(), np.abs(psi2)**2, '-', label='2', c='y')
        ax2.set_ylabel('Probability Density')
        ax22 = ax2.twinx()
        ax22.plot(basis(), basis(v2), '--', c='grey')
        ax22.set_ylabel('Potential')
        ax22.set_ylim(0, 5)
        ax2.legend()

        plt.savefig('q3-3/wfn{:08.0f}.png'.format(t / dt), dpi=300)
        plt.close()
    np.savetxt('q3-3.txt', np.array(zipped), header='Time S_12(a) S_12(b) Energy')



def plotter(fname):
    zipped_data = np.loadtxt(fname, dtype=DTYPE)
    head = fname.split('.')[0]
    time, s12a, s12b, energy = np.transpose(zipped_data).real

    plt.plot(time, s12a)
    plt.xlabel('Time')
    plt.ylabel(r'''$S_{12}$''')
    plt.savefig('{}_time_s12a.pdf'.format(head))
    plt.close()

    plt.plot(time, s12b)
    plt.xlabel('Time')
    plt.ylabel(r'''$S_{12}$''')
    plt.savefig('{}_time_s12b.pdf'.format(head))
    plt.close()

    plt.plot(time, energy)
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.savefig('{}_time_energy.pdf'.format(head))
    plt.close()
    return


def main():
    q3_2()
    # plotter('q3-2.txt')
    q3_3()
    # plotter('q3-3.txt')


if __name__ == '__main__':
    main()
