# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simple LCDM models'''

import numpy as np
from dataclasses import dataclass, field
from ._numerical import antideriv
from functools import lru_cache


@dataclass(frozen=True)
class LCDM:
    '''LambdaCDM background cosmology'''

    h: float
    '''dimensionless Hubble parameter'''

    Om: float
    '''matter density parameter'''

    Ol: float = None
    '''Lambda density parameter'''

    Ok: float = None
    '''curvature density parameter'''

    zmax: float = field(default=1100., repr=False)
    '''maximum redshift for automatic interpolation'''

    gamma: float = field(default=6/11, repr=False)
    '''growth rate parameter'''

    def _set(self, a, v):
        object.__setattr__(self, a, v)

    def __post_init__(self):
        # dark energy density
        if self.Ol is None:
            self._set('Ol', 1 - self.Om - (self.Ok or 0.))

        # curvature density
        if self.Ok is None:
            self._set('Ok', 1 - self.Om - self.Ol)

        # check that sum of parameters is unity
        if not np.isclose(self.Om + self.Ol + self.Ok, 1.):
            raise ValueError('density parameters do not sum to unity')

        # set the curvature parameter K = sqrt(|Ok|)
        self._set('K', np.sqrt(np.fabs(self.Ok)))

    @property
    @lru_cache(maxsize=None)
    def _xc(self):
        '''dimensionless comoving distance interpolator'''
        def f(z):
            return 1/self.e(z)
        return antideriv(f, 0., self.zmax, 0.01, inverse=True)

    @property
    @lru_cache(maxsize=None)
    def _ln_gf(self):
        '''logarithm of growth function interpolator'''
        def f(z):
            return -1/(1 + z) * self.Omz(z)**self.gamma
        return antideriv(f, 0., self.zmax, 0.01)

    @property
    @lru_cache(maxsize=None)
    def dh(self):
        '''Hubble distance'''
        return 2997.92458/self.h

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
        '''redshift-dependent Lambda density parameter'''
        return self.Ol/self.Oz(z)

    def Okz(self, z):
        '''redshift-dependent curvature density parameter'''
        return self.Ok*(1+z)**2/self.Oz(z)

    def e(self, z):
        '''dimensionless Hubble function'''
        return np.sqrt(self.Oz(z))

    def gf(self, z):
        '''growth function'''
        return np.exp(self._ln_gf(z))

    def xc(self, z, zp=None):
        '''dimensionless comoving distance'''
        if zp is None:
            return self._xc[0](z)
        else:
            return self.xc(zp) - self.xc(z)

    def dc(self, z, zp=None):
        '''comoving distance'''
        return self.dh * self.xc(z, zp)

    def xc_inv(self, xc):
        '''inverse function for dimensionless comoving distance'''
        return self._xc[1](xc)

    def dc_inv(self, dc):
        '''inverse function for comoving distance'''
        return self.xc_inv(dc/self.dh)

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

    def dvc(self, z):
        r'''dimensionless differential comoving volume

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function returns

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)**2}}{\mathtt{e(z)}} \;.

        '''

        return self.xm(z)**2/self.e(z)

    def vc(self, z, zp=None):
        '''dimensionless comoving volume'''

        if zp is not None:
            return self.vc(zp) - self.vc(z)

        x = self.xc(z)

        if self.Ok > 0:
            return (np.sinh(2*self.K*x) - 2*self.K*x)/(4*self.K**3)
        elif self.Ok < 0:
            return (2*self.K*x - np.sin(2*self.K*x))/(4*self.K**3)
        else:
            return x**3/3
