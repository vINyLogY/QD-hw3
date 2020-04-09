#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip

import numpy as np
from scipy import linalg, integrate, interpolate, fftpack

DTYPE = np.complex128


class GridBasis(object):
    def __init__(self, xmin, xmax, npts):
        self.xspace = np.linspace(xmin, xmax, num=npts,
                                  endpoint=True, dtype=DTYPE)
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
    def width(self):
        return self[-1] - self[0]

    def inner_product(self, f1, f2):
        """
        Args:
            f1: float -> float
            f2: float -> float
        """
        return integrate.simps(np.conj(self(f1)) * self(f2), self())

    def norm(self, f):
        return np.sqrt(self.inner_product(f, f))

    @staticmethod
    def mop(v):
        """Generate the operator to multiply `v(x)` to the wavefunction
        `psi(x)`.
        Args:
            v: (float -> float) -> (float -> float)
        """
        return lambda f: lambda x: v(x) * f(x)


class Coordinate(GridBasis):
    @property
    def conj(self):
        """Return the conjugate basis in momentum space from coordinate space.
        """
        length = len(self)
        width = 2 * length * np.pi / (self.width)
        half = 0.5 if length % 2 else 0.5 * (1 - 1.0 / (length - 1))
        return Momentum((half - 1) * width, half * width, length)

    def ft(self, f):
        coeff = np.sqrt(2.0 * np.pi) / (self.conj[-1] - self.conj[0])
        sample = (coeff *
                  fftpack.fftshift(fftpack.fft(fftpack.ifftshift(self(f)))))
        return interpolate.interp1d(self.conj(), sample)


class Momentum(GridBasis):
    @property
    def conj(self):
        """Return the conjugate basis in coordinate space from momentum space.
        """
        length = len(self)
        half_width = length * np.pi / (self.width)
        return Coordinate(-half_width, half_width, length)

    def ft(self, f):
        coeff = (np.sqrt(2.0 * np.pi) / (self.conj.width) *
                 len(self))
        sample = (coeff *
                  fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(self(f)))))
        return interpolate.interp1d(self.conj(), sample)


class Propagator(object):
    def __init__(self, init_wfn, basis, t, v):
        self.wfn = init_wfn
        self.basis = basis
        self.t = t
        self.v = v

        self.renormalization_coeff = 1.0

    def __call__(self, dt=0.1, max_iter=1e4, renormalization_level=0):
        op = GridBasis.mop
        exp = np.exp
        basis = self.basis
        def uv(tau): return lambda x: exp(-1.0j * self.v(x) * tau)
        def ut(tau): return lambda p: exp(-1.0j * self.t(p) * tau)

        yield (0.0, self.wfn)
        for i in range(max_iter):
            self.wfn = op(uv(dt / 2.0))(self.wfn)
            if renormalization_level // 2 % 2:
                self.renormalization()
            self.wfn = basis.ft(self.wfn)
            self.wfn = op(ut(dt))(self.wfn)
            if renormalization_level // 4 % 2:
                self.renormalization(conj=True)
            self.wfn = basis.conj.ft(self.wfn)
            self.wfn = op(uv(dt / 2.0))(self.wfn)
            if renormalization_level % 2:
                self.renormalization()
            yield ((i + 1) * dt, self.wfn)

    def renormalization(self, conj=False):
        basis = self.basis.conj if conj else self.basis
        wfn = self.wfn
        norm = basis.norm(wfn)
        self.renormalization_coeff *= norm
        self.wfn = lambda x: wfn(x) / norm
        return

def gaussian_wfn(x0=0.0, p0=0.0, alpha=1.0):
    pi = np.pi
    exp = np.exp

    def gaussian(x):
        return (alpha / pi) ** 0.25 * exp(-alpha * (x-x0)**2 / 2) * exp(1.0j * p0 * (x-x0))

    return gaussian


def test_conjconj(npts=1024):
    basis1 = Coordinate(-20.0, 20.0, npts)
    basis2 = basis1.conj.conj
    assert (basis1 == basis2)

    psi1 = gaussian_wfn()
    psi2 = basis1.conj.ft(basis1.ft(psi1))
    assert np.allclose(basis1(psi1), basis2(psi2))
    return


if __name__ == "__main__":
    test_conjconj()
