# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for background cosmology'''

from abc import ABCMeta, abstractmethod
import numpy as np


class Cosmology(metaclass=ABCMeta):
    '''Interface for cosmological background calculations.'''

    @classmethod
    def from_astropy(cls, cosmo):
        '''Construct a Cosmology instance from Astropy cosmologies.'''
        from .astropy import AstropyCosmology
        return AstropyCosmology(cosmo)

    @classmethod
    def from_camb(cls, *args):
        '''Construct a Cosmology instance from CAMB.

        Takes CAMB parameters, results, or both, and returns an instance of a
        :class:`Cosmology` adapter class.

        '''

        from .camb import CambCosmology
        return CambCosmology(*args)

    @classmethod
    def _implements(cls, name):
        '''Check whether subclass implements an abstract method.'''
        method = getattr(cls, name, None)
        if method is None:
            return False
        return not getattr(method, '__isabstractmethod__', False)

    @classmethod
    def _default_methods(cls):
        '''List available default methods.'''
        defaults = []
        for c in cls.__mro__:
            for name, value in vars(c).items():
                if name.startswith('_'):
                    parts = name[1:].split('_from_')
                    if len(parts) == 2:
                        method = parts[0]
                        requires = parts[1].split('_and_') if parts[1] else []
                        defaults.append((method, requires, value))
        return defaults

    def __init_subclass__(cls, **kwargs):
        '''Fill out a subclass with available default implementations.'''
        super().__init_subclass__(**kwargs)
        defaults = cls._default_methods()
        for method, requires, default in defaults:
            if not cls._implements(method) \
                    and all(map(cls._implements, requires)):
                setattr(cls, method, default)

    # these are default implementations of methods
    # their names are _{method}_from_{requires}_and_{requires}_and_...
    # their order matters, as they will be checked from top to bottom

    @property
    def _h0_from_hf(self) -> float:
        '''h0 = hf(0)'''
        return self.hf(0)

    @property
    def _h0_from_h(self) -> float:
        '''h0 = 100*h'''
        return self.h*100

    @property
    def _h_from_h0(self) -> float:
        '''h = h0/100'''
        return self.h0/100

    @property
    def _dh_from_h(self) -> float:
        '''dh = 2997.92458 Mpc/h by definition'''
        return 2997.92458/self.h

    @property
    def _grav_from_(self) -> float:
        '''using CODATA 2018 value for Newton's G'''
        return 4.30091727003628e-3

    @property
    def _rho_c_from_grav_and_h(self) -> float:
        return 3e10/(8*np.pi*self.grav)*self.h**2

    def _a_from_(self, z):
        '''a = 1/(1+z)'''
        return 1/np.add(1, z)

    def _ef_from_hf_and_h0(self, z):
        return self.hf(z)/self.h0

    def _ef_from_omega_z(self, z):
        return self.omega_z(z)**0.5

    def _hf_from_ef_and_h0(self, z):
        return self.ef(z)*self.h0

    def _omega_z_from_ef(self, z):
        return self.ef(z)**2

    def _omega_m_z_from_omega_z(self, z):
        return self.omega_m*np.add(1, z)**3/self.omega_z(z)

    def _omega_k_z_from_omega_z(self, z):
        return self.omega_k*np.add(1, z)**2/self.omega_z(z)

    def _omega_de_z_from_omega_m_z_and_omega_k_z(self, z):
        return 1 - self.omega_m_z(z) - self.omega_k_z(z)

    def _rho_c_z_from_rho_c_and_omega_z(self, z):
        return self.rho_c*self.omega_z(z)

    def _rho_m_z_from_rho_c_z_and_omega_m_z(self, z):
        return self.rho_c_z(z)*self.omega_m_z(z)

    def _rho_de_z_from_rho_c_z_and_omega_de_z(self, z):
        return self.rho_c_z(z)*self.omega_de_z(z)

    def _rho_k_z_from_rho_c_z_and_omega_k_z(self, z):
        return self.rho_c_z(z)*self.omega_k_z(z)

    def _dc_from_xc_and_dh(self, z, zp=None):
        return self.xc(z, zp)*self.dh

    def _xc_from_dc_and_dh(self, z, zp=None):
        return self.dc(z, zp)/self.dh

    def _dc_inv_from_xc_inv_and_dh(self, dc):
        return self.xc_inv(dc/self.dh)

    def _xc_inv_from_dc_inv_and_dh(self, xc):
        return self.dc_inv(xc*self.dh)

    def _dm_from_xm_and_dh(self, z, zp=None):
        return self.xm(z, zp)*self.dh

    def _dm_from_da(self, z, zp=None):
        return self.da(z, zp)*np.add(1, z if zp is None else zp)

    def _xm_from_xa(self, z, zp=None):
        return self.xa(z, zp)*np.add(1, z if zp is None else zp)

    def _xm_from_dm_and_dh(self, z, zp=None):
        return self.dm(z, zp)/self.dh

    def _dm_from_dc_and_dh(self, z, zp=None):
        if self.omega_k == 0:
            return self.dc(z, zp)
        if self.omega_k > 0:
            k = self.omega_k**0.5
            return np.sinh(self.dc(z, zp)/self.dh*k)/k*self.dh
        else:
            k = (-self.omega_k)**0.5
            return np.sin(self.dc(z, zp)/self.dh*k)/k*self.dh

    def _xm_from_xc(self, z, zp=None):
        if self.omega_k == 0:
            return self.xc(z, zp)
        if self.omega_k > 0:
            k = self.omega_k**0.5
            return np.sinh(self.xc(z, zp)*k)/k
        else:
            k = (-self.omega_k)**0.5
            return np.sin(self.xc(z, zp)*k)/k

    def _xm_inv_from_dm_inv_and_dh(self, xm):
        return self.dm_inv(xm*self.dh)

    def _dm_inv_from_xm_inv_and_dh(self, dm):
        return self.xm_inv(dm/self.dh)

    def _dm_inv_from_dc_inv_and_dh(self, dm):
        if self.omega_k == 0:
            return self.dc_inv(dm)
        elif self.omega_k > 0:
            k = self.omega_k**0.5
            return self.dc_inv(np.arcsinh(dm/self.dh*k)/k*self.dh)
        else:
            k = (-self.omega_k)**0.5
            return self.dc_inv(np.arcsin(dm/self.dh*k)/k*self.dh)

    def _xm_inv_from_xc_inv(self, xm):
        if self.omega_k == 0:
            return self.xc_inv(xm)
        elif self.omega_k > 0:
            k = self.omega_k**0.5
            return self.xc_inv(np.arcsinh(xm*k)/k)
        else:
            k = (-self.omega_k)**0.5
            return self.xc_inv(np.arcsin(xm*k)/k)

    def _da_from_xa_and_dh(self, z, zp=None):
        return self.xa(z, zp)*self.dh

    def _xa_from_da_and_dh(self, z, zp=None):
        return self.da(z, zp)/self.dh

    def _da_from_dm(self, z, zp=None):
        return self.dm(z, zp)/np.add(1, z if zp is None else zp)

    def _xa_from_xm(self, z, zp=None):
        return self.xm(z, zp)/np.add(1, z if zp is None else zp)

    def _da_inv_from_xa_inv_and_dh(self, da):
        return self.xa_inv(da/self.dh)

    def _da_inv_from_(self, da):
        '''raises NotImplementedError'''
        raise NotImplementedError

    def _xa_inv_from_da_inv_and_dh(self, xa):
        return self.da_inv(xa*self.dh)

    def _dvc_from_xm_and_ef(self, z):
        return self.xm(z)**2/self.ef(z)

    def _vc_from_xc(self, z, zp=None):
        if zp is not None:
            return self.vc(zp) - self.vc(z)
        x = self.xc(z)
        if self.omega_k == 0:
            return x**3/3
        elif self.omega_k > 0:
            k = self.omega_k**0.5
            return (np.sinh(2*k*x) - 2*k*x)/(4*k**3)
        else:
            k = (-self.omega_k)**0.5
            return (2*k*x - np.sin(2*k*x))/(4*k**3)

    # end of default implementations

    @property
    @abstractmethod
    def h(self) -> float:
        '''Dimensionless Hubble parameter :math:`h = H_0/(100 \\, \\rm{km} \\,
        \\rm{s}^{-1} \\, \\rm{Mpc}^{-1})`.'''
        pass

    @property
    @abstractmethod
    def h0(self) -> float:
        '''Hubble function at redshift 0 in km s-1 Mpc-1.'''
        pass

    @property
    @abstractmethod
    def dh(self) -> float:
        '''Hubble distance in Mpc.'''
        pass

    @property
    @abstractmethod
    def omega_m(self) -> float:
        '''Matter density parameter at redshift 0.'''
        pass

    @property
    @abstractmethod
    def omega_de(self) -> float:
        '''Dark energy density parameter at redshift 0.'''
        pass

    @property
    @abstractmethod
    def omega_k(self) -> float:
        '''Curvature density parameter at redshift 0.'''
        pass

    @property
    @abstractmethod
    def rho_c(self) -> float:
        '''Critical density at redshift 0 in Msol Mpc-3.'''
        pass

    @property
    @abstractmethod
    def grav(self) -> float:
        '''Gravitational constant G in pc km2 s-2 Msol-1.'''
        pass

    @abstractmethod
    def a(self, z):
        '''Redshift-dependent scale factor.'''
        pass

    @abstractmethod
    def hf(self, z):
        '''Hubble function :math:`H(z)` in km s-1 Mpc-1.'''
        pass

    @abstractmethod
    def ef(self, z):
        '''Standardised Hubble function :math:`E(z) = H(z)/H_0`.'''
        pass

    @abstractmethod
    def omega_z(self, z):
        '''Redshift-dependent total density parameter.'''
        pass

    @abstractmethod
    def omega_m_z(self, z):
        '''Redshift-dependent matter density parameter.'''
        pass

    @abstractmethod
    def omega_de_z(self, z):
        '''Redshift-dependent dark energy density parameter.'''
        pass

    @abstractmethod
    def omega_k_z(self, z):
        '''Redshift-dependent curvature density parameter.'''
        pass

    @abstractmethod
    def rho_c_z(self, z):
        '''Redshift-dependent critical density in Msol Mpc-3.'''
        pass

    @abstractmethod
    def rho_m_z(self, z):
        '''Redshift-dependent matter density in Msol Mpc-3.'''
        pass

    @abstractmethod
    def rho_de_z(self, z):
        '''Redshift-dependent dark energy density in Msol Mpc-3.'''
        pass

    @abstractmethod
    def rho_k_z(self, z):
        '''Redshift-dependent curvature density in Msol Mpc-3.'''
        pass

    @abstractmethod
    def dc(self, z, zp=None):
        '''Comoving distance :math:`d_c(z)` in Mpc.'''
        pass

    @abstractmethod
    def xc(self, z, zp=None):
        '''Dimensionless comoving distance :math:`x_c'(z) = d_c(z)/d_H`.'''
        pass

    @abstractmethod
    def dc_inv(self, dc):
        '''Inverse function for the comoving distance in Mpc.'''
        pass

    @abstractmethod
    def xc_inv(self, xc):
        '''Inverse function for the dimensionless comoving distance.'''
        pass

    @abstractmethod
    def dm(self, z, zp=None):
        '''Transverse comoving distance :math:`d_M(z)` in Mpc.'''
        pass

    @abstractmethod
    def xm(self, z, zp=None):
        '''Dimensionless transverse comoving distance :math:`x_M(z) =
        d_M(z)/d_H`.'''
        pass

    @abstractmethod
    def dm_inv(self, dm):
        '''Inverse function for the transverse comoving distance in Mpc.'''
        pass

    @abstractmethod
    def xm_inv(self, xm):
        '''Inverse function for the dimensionless transverse comoving
        distance.'''
        pass

    @abstractmethod
    def da(self, z, zp=None):
        '''Angular diameter distance :math:`d_A(z)` in Mpc.'''
        pass

    @abstractmethod
    def xa(self, z, zp=None):
        '''Dimensionless angular diameter distance :math:`x_A(z) =
        d_A(z)/d_H`.'''
        pass

    @abstractmethod
    def da_inv(self, da):
        '''Inverse function for the angular diameter distance in Mpc.'''
        pass

    @abstractmethod
    def xa_inv(self, xa):
        '''Inverse function for the dimensionless angular diameter distance.'''
        pass

    @abstractmethod
    def dvc(self, z):
        r'''Dimensionless differential comoving volume.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function returns

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        '''
        pass

    @abstractmethod
    def vc(self, z, zp=None):
        '''Dimensionless comoving volume.'''
        pass
