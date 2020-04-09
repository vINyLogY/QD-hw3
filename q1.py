#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc

from grids import Coordinate, gaussian_wfn

rc('font', family='Times New Roman')
rc('text', usetex=True)

# always set hbar = 1.0
op = Coordinate.mop


def q1_1(npts):
    print('{} grid points:'.format(npts))
    basis = Coordinate(-20.0, 20.0, npts)
    psi = gaussian_wfn()
    norm = basis.norm(psi)
    x = basis.inner_product(psi, op(lambda x: x)(psi))
    x2 = basis.inner_product(psi, op(lambda x: x**2)(psi))
    delta_x = np.sqrt(x2 - x**2)
    print('[N] Norm: {:.8f}; <x>: {:.8f}; Delta x: {:.8f}'.format(
        norm, x, delta_x))
    print('[A] Norm: 1; <x>: 0; Delta x: {:.8f}'.format(1.0 / np.sqrt(2)))

    plt.plot(basis(), basis(psi).real, '.', label='Re')
    plt.plot(basis(), basis(psi).imag, '--', label="Im")
    plt.legend()
    plt.show()
    return


def q1_2(npts):
    print('{} grid points:'.format(npts))
    basis = Coordinate(-20.0, 20.0, npts)
    p_basis = basis.conj
    psi = gaussian_wfn()
    psi_p = basis.ft(psi)
    norm = p_basis.norm(psi_p)
    p = p_basis.inner_product(psi_p, op(lambda x: x)(psi_p))
    p2 = p_basis.inner_product(psi_p, op(lambda x: x**2)(psi_p))
    delta_p = np.sqrt(p2 - p**2)
    print('[N] Norm: {:.8f}; <p>: {:.8f}; Delta p: {:.8f}'.format(
        norm, p, delta_p))
    print('[A] Norm: 1; <p>: 0; Delta p: {:.8f}'.format(1.0 / np.sqrt(2)))

    plt.plot(p_basis(), p_basis(psi_p).real, '.', label='Re')
    plt.plot(p_basis(), p_basis(psi_p).imag, '--', label="Im")
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()
    return


def q1_3(npts):
    omega = 1.0
    print('omega = {:.2f}, {} grid points:'.format(omega, npts))
    basis = Coordinate(-20.0, 20.0, npts)
    p_basis = basis.conj
    psi = gaussian_wfn()
    psi_p = basis.ft(psi)
    v = op(lambda x: 0.5 * omega**2 * x**2)
    t = op(lambda p: 0.5 * p**2)
    potential = basis.inner_product(psi, v(psi))
    kinetic = p_basis.inner_product(psi_p, t(psi_p))
    total = potential + kinetic
    print('[N] Potential: {:.8f}; Kinetic: {:.8f}; Total: {:.8f}'.format(
        potential, kinetic, total))
    print('[A] ZPE = {:.8f}'.format(0.5 * omega))
    return


def main():
    npts = 1024
    q1_1(npts)
    q1_2(npts)
    q1_3(npts)
    return


if __name__ == '__main__':
    main()
