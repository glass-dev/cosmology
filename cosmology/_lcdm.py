# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simple LCDM models'''

import numpy as np
from dataclasses import dataclass, field
from ._numerical import antideriv


@dataclass(frozen=True)
class LCDM:
    H0: float
    Om: float
    Ol: float = None
    Ok: float = None
    dh: float = field(init=False, repr=False)
    zmax: float = field(default=1100., repr=False)
    gamma: float = field(default=6/11, repr=False)

    def __post_init__(self):
        # dark energy density
        if self.Ol is None:
            object.__setattr__(self, 'Ol', 1 - self.Om - (self.Ok or 0.))

        # curvature density
        if self.Ok is None:
            object.__setattr__(self, 'Ok', 1 - self.Om - self.Ol)

        # check that sum of parameters is unity
        if not np.isclose(self.Om + self.Ol + self.Ok, 1.):
            raise ValueError('density parameters do not sum to unity')

        # set the curvature parameter K = sqrt(|Ok|)
        object.__setattr__(self, 'K', np.sqrt(np.fabs(self.Ok)))

        # Hubble distance
        object.__setattr__(self, 'dh', 299792.458/self.H0)

        # compute the comoving distance interpolator
        _xc, _xc_inv = antideriv(lambda z: 1/self.e(z), 0., self.zmax, 0.01, inverse=True)
        object.__setattr__(self, '_xc', _xc)
        object.__setattr__(self, '_xc_inv', _xc_inv)

        # compute the growth factor interpolator
        _gf = antideriv(lambda z: self.Omz(z)**self.gamma/(1 + z), 0., self.zmax, 0.01)
        object.__setattr__(self, '_gf', _gf)

    def a(self, z):
        '''scale factor'''
        return 1/np.add(z, 1)

    def Oz(self, z):
        '''redshift-dependent total density parameter'''
        zp1 = np.add(z, 1)
        return self.Om*zp1**3 + self.Ol + self.Ok*zp1**2

    def Omz(self, z):
        '''redshift-dependent matter density parameter'''
        return self.Om*(1+z)**3/self.Oz(z)

    def Olz(self, z):
        '''redshift-dependent lambda density parameter'''
        return self.Ol/self.Oz(z)

    def Okz(self, z):
        '''redshift-dependent curvature density parameter'''
        return self.Ok*(1+z)**2/self.Oz(z)

    def e(self, z):
        '''dimensionless Hubble function'''
        return np.sqrt(self.Oz(z))

    def gf(self, z):
        '''growth function'''
        return np.exp(-self._gf(z))

    def xc(self, z, zp=None):
        '''dimensionless comoving distance'''
        if zp is None:
            return self._xc(z)
        else:
            return self.xc(zp) - self.xc(z)

    def dc(self, z, zp=None):
        '''comoving distance'''
        return self.dh * self.xc(z, zp)

    def xm(self, z, zp=None):
        '''dimensionless transverse comoving distance'''
        if self.Ok > 0:
            return np.sinh(self.K * self.xc(z, zp))/self.K
        elif self.Ok < 0:
            return np.sin(self.K * self.xc(z, zp))/self.K
        else:
            return self.xc(z, zp)

    def dm(self, z, zp=None):
        '''transverse comoving distance'''
        return self.dh * self.xm(z, zp)

    def xc_inv(self, xc):
        '''inverse function for dimensionless comoving distance'''
        return self._xc_inv(xc)

    def dc_inv(self, dc):
        '''inverse function for comoving distance'''
        return self.xc_inv(dc/self.dh)

    def xm_inv(self, xm):
        '''inverse function for dimensionless transverse comoving distance'''
        if self.Ok > 0:
            return self.xc_inv(np.arcsinh(self.K * xm)/self.K)
        elif self.Ok < 0:
            return self.xc_inv(np.arcsin(self.K * xm)/self.K)
        else:
            return self.xc_inv(xm)

    def dm_inv(self, dm):
        '''inverse function for transverse comoving distance'''
        return self.xm_inv(dm/self.dh)
