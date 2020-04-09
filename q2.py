#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rc

from grids import Coordinate, gaussian_wfn, Propagator
from minitn.bases.dvr import SineDVR

rc('font', family='Times New Roman')
rc('text', usetex=True)

# always set hbar = 1.0
op = Coordinate.mop
DTYPE = np.complex128


def q2_1(npts, max_iter=1000):
    init_wfn = gaussian_wfn(x0=-2.5)
    basis = Coordinate(-20, 20, npts)
    def v_func(x): return 0.5 * x**2
    def t_func(p): return 0.5 * p**2
    propergator = Propagator(init_wfn, basis, t_func, v_func)
    dt = 0.1
    zipped = []
    for t, wfn in propergator(dt=dt, max_iter=max_iter):
        print('Time: {:.2f}'.format(t))
        norm = basis.norm(wfn)
        x = basis.inner_product(wfn, op(lambda x: x)(wfn))
        x2 = basis.inner_product(wfn, op(lambda x: x**2)(wfn))
        delta_x = np.sqrt(x2 - x ** 2)
        print('[Q] Norm: {:.8f}; <x>: {:.8f}; Delta x: {:.8f}'.format(
            norm, x, delta_x))
        wfn_p = basis.ft(wfn)
        basis_p = basis.conj
        norm_p = basis_p.norm(wfn_p)
        p = basis_p.inner_product(wfn_p, op(lambda x: x)(wfn_p))
        p2 = basis_p.inner_product(wfn_p, op(lambda x: x**2)(wfn_p))
        delta_p = np.sqrt(p2 - p**2)
        print('[P] Norm: {:.8f}; <p>: {:.8f}; Delta p: {:.8f}'.format(
            norm_p, p, delta_p))
        energy = 0.5 * p2 + 0.5 * x2
        print('Energy: {:.8f}'.format(energy))
        zipped.append((t, x, p, energy))
        plt.plot(basis(), basis(wfn).real, '-', label='Re')
        plt.plot(basis(), basis(wfn).imag, '--', label="Im")
        plt.xlim(-20, 20)
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig('q2-1/wfn{:08.0f}.png'.format(t / dt), dpi=300)
        plt.close()
    np.savetxt('q2-1.txt', np.array(zipped),
               header='Time Position Momentum Energy')
    return


def q2_1a(max_iter=1000):
    dt = 0.1
    zipped = []
    for i in range(max_iter + 1):
        zipped.append((
            dt*i,
            -2.5*np.cos(dt*i),
            2.5*np.sin(dt*i),
            0.5 * 2.5 ** 2 + 0.5
        ))
    return np.array(zipped)


def q2_2(npts, max_iter=1000):
    init_wfn = gaussian_wfn(x0=0.5)
    basis = Coordinate(-20, 20, npts)
    def v_func(x): return 8 * (1.0 - np.exp(-x/4))**2
    def t_func(p): return 0.5 * p**2
    propergator = Propagator(init_wfn, basis, t_func, v_func)
    dt = 0.1
    zipped = []
    for t, wfn in propergator(dt=dt, max_iter=max_iter):
        print('Time: {:.2f}'.format(t))
        norm = basis.norm(wfn)
        x = basis.inner_product(wfn, op(lambda x: x)(wfn))
        potential = basis.inner_product(wfn, op(v_func)(wfn))
        print('[Q] Norm: {:.8f}; <x>: {:.8f}; Potential: {:.8f}'.format(
            norm, x, potential))
        wfn_p = basis.ft(wfn)
        basis_p = basis.conj
        norm_p = basis_p.norm(wfn_p)
        p = basis_p.inner_product(wfn_p, op(lambda x: x)(wfn_p))
        kinetic = basis_p.inner_product(wfn_p, op(t_func)(wfn_p))
        print('[P] Norm: {:.8f}; <p>: {:.8f}; Kinetic: {:.8f}'.format(
            norm_p, p, kinetic))
        energy = kinetic + potential
        print('Energy: {:.8f}'.format(energy))
        zipped.append((t, x, p, energy))
        plt.plot(basis(), basis(wfn).real, '-', label='Re')
        plt.plot(basis(), basis(wfn).imag, '--', label="Im")
        plt.xlim(-20, 20)
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig('q2-2/wfn{:08.0f}.png'.format(t / dt), dpi=300)
        plt.close()
    np.savetxt('q2-2.txt', np.array(zipped),
               header='Time Position Momentum Energy')
    return


def q2_3(npts, max_iter=1000, err=1.0e-14):
    init_wfn = gaussian_wfn(x0=0.5)
    basis = Coordinate(-20, 20, npts)
    def v_func(x): return 8 * (1.0 - np.exp(-x/4))**2
    def t_func(p): return 0.5 * p**2
    propergator = Propagator(init_wfn, basis, t_func, v_func)
    dt = -0.1j
    prev_energy = None
    for t, wfn in propergator(dt=dt, max_iter=max_iter,
                              renormalization_level=1):
        potential = basis.inner_product(wfn, op(v_func)(wfn))
        wfn_p = basis.ft(wfn)
        kinetic = basis.conj.inner_product(wfn_p, op(t_func)(wfn_p))
        energy = kinetic + potential
        norm = propergator.renormalization_coeff
        print('Time: {:.2f}; Norm: {:.8f}; Energy: {:.8f}'.format(
            t, norm, energy))

        plt.plot(basis(), basis(wfn).real, '-', label='Re')
        plt.plot(basis(), basis(wfn).imag, '--', label="Im")
        plt.xlim(-20, 20)
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig('q2-3/wfn{:08.0f}.png'.format(np.abs(t / dt)), dpi=300)
        plt.close()
        if prev_energy and np.allclose(prev_energy, energy, atol=err, rtol=err):
            break
        else:
            prev_energy = energy
    
    print('SO Energy: {:.8f}'.format(energy))
    zipped = np.array((basis(), basis(propergator.wfn)))
    np.savetxt('q2-3.txt', zipped)
    return


def q2_3dvr(npts):
    from minitn.bases.dvr import FastSineDVR
    def v_func(x): return 8 * (1.0 - np.exp(-x/4))**2
    solver = FastSineDVR(-20, 20, npts)
    solver.set_v_func(v_func)
    energy, v = solver.solve(n_state=1)
    print('DVR Energy: {:.8f}'.format(energy[0]))
    
    dvr_wfn = solver.dvr2cont(v[0])
    basis, wfn = np.loadtxt('q2-3.txt', dtype=DTYPE)
    
    xspace = np.linspace(-20, 20, num=101)
    plt.plot(xspace, dvr_wfn(xspace).real, '.', label='DVR Re')
    plt.plot(xspace, dvr_wfn(xspace).imag, '.', label='DVR Im')
    plt.plot(basis, wfn.real, '-', label='SO Re')
    plt.plot(basis, wfn.imag, '--', label='SO Im')
    plt.xlim(-20, 20)
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig('q2-3.pdf')
    plt.close()
    return


def q2_4(npts, max_iter=1000):
    init_wfn = gaussian_wfn(x0=-5.5, p0=2.0)
    basis = Coordinate(-20, 20, npts)
    def v_func(x): return np.where(np.abs(x) < 0.5, 3.0, 0.0)

    def va_func(x):
        return np.where(np.abs(x) > 10.0, -1.0j * (np.abs(x) - 10.0)**4, 0.0)

    def t_func(p): return 0.5 * p ** 2

    def _v(x): return v_func(x) + va_func(x)
    propergator = Propagator(init_wfn, basis, t_func, _v)
    dt = 0.1
    for t, wfn in propergator(dt=dt, max_iter=max_iter,
                              renormalization_level=1):
        norm = propergator.renormalization_coeff
        print('Time: {:.2f}; Norm: {:.8f}'.format(t, norm))
        plt.plot(basis(), norm * basis(wfn).real, '-', label='Re')
        plt.plot(basis(), norm * basis(wfn).imag, '--', label="Im")
        plt.xlim(-20, 20)
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig('q2-4/wfn{:08.0f}.png'.format(t / dt), dpi=300)
        plt.close()
    return


def plotter(fname, x_range, p_range, e_range, ref=None):
    zipped_data = np.loadtxt(fname, dtype=DTYPE)
    head = fname.split('.')[0]
    time, position, momentum, energy = np.transpose(zipped_data).real
    if ref is not None:
        rtime, rposition, rmomentum, renergy = np.transpose(ref).real

    plt.plot(time, position, label='SO')
    if ref is not None:
        plt.plot(rtime, rposition, '--', label='Analytic')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.ylim(*x_range)
    plt.savefig('{}_time_position.pdf'.format(head))
    plt.close()

    plt.plot(time, momentum, label='SO')
    if ref is not None:
        plt.plot(rtime, rmomentum, '--', label='Analytic')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    plt.ylim(*p_range)
    plt.savefig('{}_time_momentum.pdf'.format(head))
    plt.close()

    plt.plot(time, energy, label='SO')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    if ref is not None:
        plt.plot(rtime, renergy, '--', label='Analytic')
        plt.legend()
    plt.ylim(e_range)
    plt.savefig('{}_time_energy.pdf'.format(head))
    plt.close()
    return


def main():
    npts = 2 ** 12
    # 2-1
    # q2_1(npts)
    # plotter('q2-1.txt', (-3, 3), (-3, 3), (3.61, 3.63), ref=q2_1a())

    # 2-2
    # q2_2(npts)
    # plotter('q2-2.txt', (-0.25, 0.75), (-0.5, 0.5), (0.53, 0.55))

    # 2-3
    # q2_3(npts)
    q2_3dvr(1024)

    # 2-4
    # q2_4(npts)

    return


if __name__ == '__main__':
    main()
