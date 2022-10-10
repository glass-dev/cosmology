# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for CAMB integration'''

from numbers import Number
import numpy as np

import camb

from .background import Cosmology


def _zarg(*zs):
    za = [z if z is None or isinstance(z, Number) else np.asanyarray(z)
          for z in zs]
    return za if len(za) > 1 else za[0]


class CambCosmology(Cosmology):

    def __init__(self, *args):
        self._p = self._r = None
        for arg in args:
            if isinstance(arg, camb.CAMBparams):
                self._p = arg
            elif isinstance(arg, camb.CAMBdata):
                self._r = arg
            else:
                raise TypeError('argument not CAMB parameters or results')
        if self._p is None and self._r is None:
            raise ValueError('requires CAMB parameters and/or results')
        if self._p is None:
            self._p = self._r.Params
        if self._r is None:
            self._r = camb.get_background(self._p)

    @property
    def h(self) -> float:
        return self._p.h

    @property
    def h0(self) -> float:
        return self._p.H0

    @property
    def omega_m(self) -> float:
        return self._p.omegam

    @property
    def omega_de(self) -> float:
        return self._r.omega_de

    @property
    def omega_k(self) -> float:
        return self._p.omk

    def hf(self, z):
        z = _zarg(z)
        return self._r.hubble_parameter(z)

    def omega_m_z(self, z):
        z = _zarg(z)
        return (self._r.get_Omega('baryon', z) + self._r.get_Omega('cdm', z)
                + self._r.get_Omega('nu', z))

    def omega_de_z(self, z):
        z = _zarg(z)
        return self._r.get_Omega('de', z)

    def omega_k_z(self, z):
        z = _zarg(z)
        return self._r.get_Omega('K', z)

    def dc(self, z, zp=None):
        z, zp = _zarg(z, zp)
        if zp is not None:
            return self._r.comoving_radial_distance(zp) \
                    - self._r.comoving_radial_distance(z)
        return self._r.comoving_radial_distance(z)

    def dc_inv(self, dc):
        return self._r.redshift_at_comoving_radial_distance(dc)

    def da(self, z, zp=None):
        z, zp = _zarg(z, zp)
        if zp is not None:
            return self._r.angular_diameter_distance2(z, zp)
        return self._r.angular_diameter_distance(z)
