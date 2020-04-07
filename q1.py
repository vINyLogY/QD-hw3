#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg, integrate, interpolate
from matplotlib import pyplot as plt
from matplotlib import rc

rc('font', family='Times New Roman')
rc('text', usetex=True)

DTYPE = np.complex128
# always set hbar = 1.0

fftpack = np.fft


def gaussian_wfn(x0=0.0, p0=0.0, alpha=1.0):
    pi = np.pi
    exp = np.exp

    def _psi(x):
        return (alpha / pi) ** 0.25 * exp(-alpha * (x-x0)**2 / 2) * exp(-1.0j * p0 * (x-x0))

    return _psi


class GridBasis(object):
    def __init__(self, xmin, xmax, npts):
        self.xspace = np.linspace(
            xmin, xmax, num=npts, endpoint=True, dtype=DTYPE)
        return

    def __len__(self):
        return len(self.xspace)

    def __call__(self, f=None):
        return f(self.xspace) if f is not None else self.xspace

    def __getitem__(self, key):
        return self.xspace[key]

    def __eq__(self, other):
        return np.allclose(self(), other())

    @property
    def dx(self):
        return (self[-1] - self[0]) / (len(self.xspace) - 1)

    @staticmethod
    def mop(v):
        """Generate the operator to multiply `v(x)` to the wavefunction
        `psi(x)`.
        Args:
            v: (float -> float) -> (float -> float)
        """
        return lambda f: (lambda x: v(x) * f(x))

    def inner_product(self, f1, f2):
        """
        Args:
            f1: float -> float
            f2: float -> float
        """
        return integrate.simps(np.conj(self(f1)) * self(f2), self())

    def norm(self, f):
        return np.sqrt(self.inner_product(f, f))

    @property
    def conj(self):
        length = len(self)
        width = 2 * length * np.pi / (self[-1] - self[0])
        half = 0.5 if length % 2 else 0.5 * (1 - 1.0 / (length - 1))
        return type(self)((half - 1) * width, half * width, length)

    def fft(self, f):
        coeff = np.sqrt(2.0 * np.pi) / (self.conj[-1] - self.conj[0])
        sample = (coeff *
                  fftpack.fftshift(fftpack.fft(fftpack.ifftshift(self(f)))))
        return interpolate.interp1d(self.conj(), sample, assume_sorted=True)

    def ifft(self, f):
        coeff = (np.sqrt(2.0 * np.pi) / (self.conj[-1] - self.conj[0]) *
                 len(self))
        sample = (coeff *
                  fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(self(f)))))
        return interpolate.interp1d(self.conj(), sample, assume_sorted=True)


def q1_1(npts):
    basis = GridBasis(-20.0, 20.0, npts)
    psi = gaussian_wfn()
    op = GridBasis.mop
    norm = basis.norm(psi)
    x = basis.inner_product(psi, op(lambda x: x)(psi))
    x2 = basis.inner_product(psi, op(lambda x: x**2)(psi))
    delta_x = np.sqrt(x2 - x**2)
    print('[N] Norm: {:.8f}; <x>: {:.8f}; Delta x: {:.8f}'.format(
        norm, x, delta_x))
    print('[A] Norm: 1; <x>: 0; Delta x: {:.8f}'.format(1.0/np.sqrt(2)))
    plt.plot(basis(), basis(psi), '-')
    plt.show()


def q1_2(npts):
    print('{} grid points:'.format(npts))
    hat = GridBasis.mop
    basis = GridBasis(-20.0, 20.0, npts)
    p_basis = basis.conj
    psi = gaussian_wfn()
    psi_p = basis.fft(psi)
    norm = p_basis.norm(psi_p)
    p = p_basis.inner_product(psi_p, hat(lambda x: x)(psi_p))
    p2 = p_basis.inner_product(psi_p, hat(lambda x: x**2)(psi_p))
    delta_p = np.sqrt(p2 - p**2)
    print('[N] Norm: {:.8f}; <p>: {:.8f}; Delta p: {:.8f}'.format(
        norm, p, delta_p))
    print('[A] Norm: 1; <p>: 0; Delta p: {:.8f}'.format(1.0/np.sqrt(2)))

    plt.plot(p_basis(), p_basis(psi_p).real, '.', label='Re')
    plt.plot(p_basis(), p_basis(psi_p).imag, '--', label="Im")
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()


def q1_3(npts):
    omega = 1.0
    print('omega = {:.2f}, {} grid points:'.format(omega, npts))
    hat = GridBasis.mop
    basis = GridBasis(-20.0, 20.0, npts)
    p_basis = basis.conj
    psi = gaussian_wfn()
    psi_p = basis.fft(psi)
    v = hat(lambda x: 0.5 * omega**2 * x**2)
    t = hat(lambda p: 0.5 * p**2)

    potential = basis.inner_product(psi, v(psi))
    kinetic = p_basis.inner_product(psi_p, t(psi_p))
    total = potential + kinetic

    print('[N] Potential: {:.8f}; Kinetic: {:.8f}; Total: {:.8f}'.format(
        potential, kinetic, total))
    print('[A] ZPE = {:.8f}'.format(0.5 * omega))
    return


def main():
    # q1_1(1024)
    # q1_2(1024)
    q1_3(2**20)
    return


if __name__ == '__main__':
    main()
