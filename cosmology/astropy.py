# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for Astropy integration'''

import numpy as np
from astropy.cosmology import z_at_value
from astropy import units as u

from .background import Cosmology


class AstropyCosmology(Cosmology):

    def __init__(self, cosmology):
        self._c = cosmology

    @property
    def h(self) -> float:
        return self._c.h

    @property
    def h0(self) -> float:
        return self._c.H0.value

    @property
    def omega_m(self) -> float:
        return self._c.Om0

    @property
    def omega_de(self) -> float:
        return self._c.Ode0

    @property
    def omega_k(self) -> float:
        return self._c.Ok0

    @property
    def dh(self) -> float:
        return self._c.hubble_distance.to_value('Mpc')

    @property
    def rho_c(self) -> float:
        return self._c.critical_density0.to_value('Msun Mpc-3')

    def a(self, z):
        return self._c.scale_factor(z)

    def hf(self, z):
        return self._c.H(z).to_value('km s-1 Mpc-1')

    def ef(self, z):
        return self._c.efunc(z)

    def omega_z(self, z):
        return self._c.Otot(z)

    def omega_m_z(self, z):
        return self._c.Om(z)

    def omega_de_z(self, z):
        return self._c.Ode(z)

    def omega_k_z(self, z):
        return self._c.Ok(z)

    def rho_c_z(self, z):
        return self._c.critical_density(z).to_value('Msun Mpc-3')

    def dc(self, z, zp=None):
        if zp is not None:
            return self._c.comoving_distance(zp).value \
                    - self._c.comoving_distance(z).value
        return self._c.covoming_distance(z).value

    def dc_inv(self, dc):
        return z_at_value(self._c.comoving_distance, dc*u.Mpc).value

    def dm(self, z, zp=None):
        if zp is not None:
            return self._c.angular_diameter_distance_z1z2(z, zp).value \
                    * np.add(1, zp)
        return self._c.comoving_transverse_distance(z).value

    def dm_inv(self, dm):
        return z_at_value(self._c.comoving_transverse_distance, dm*u.Mpc).value

    def da(self, z, zp=None):
        if zp is not None:
            return self._c.angular_diameter_distance_z1z2(z, zp).value
        return self._c.angular_diameter_distance(z).value

    def da_inv(self, da):
        return z_at_value(self._c.angular_diameter_distance, da*u.Mpc).value
