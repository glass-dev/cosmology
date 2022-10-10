
=======================
Cosmological background
=======================

.. currentmodule:: cosmology

The cosmology package provides the :class:`Cosmology` class as a generic
interface (i.e. an abstract base class) to concrete implementations of
cosmological background computations.  Currently, the following adapters are
provided:

* The :func:`Cosmology.from_astropy` class method for instances of the
  :mod:`astropy.cosmology` classes.
* The :func:`Cosmology.from_camb` class method for :class:`CAMBdata`
  instances containing results from CAMB.


Cosmology interface
===================

.. autoclass:: Cosmology


Making instances
----------------

.. automethod:: Cosmology.from_astropy
.. automethod:: Cosmology.from_camb


Cosmological parameters
-----------------------

.. autoproperty:: Cosmology.h
.. autoproperty:: Cosmology.h0
.. autoproperty:: Cosmology.omega_m
.. autoproperty:: Cosmology.omega_de
.. autoproperty:: Cosmology.omega_k
.. autoproperty:: Cosmology.dh
.. autoproperty:: Cosmology.rho_c
.. autoproperty:: Cosmology.grav


Cosmology functions
-------------------

.. automethod:: Cosmology.a
.. automethod:: Cosmology.hf
.. automethod:: Cosmology.ef


Density parameters
------------------

.. automethod:: Cosmology.omega_z
.. automethod:: Cosmology.omega_m_z
.. automethod:: Cosmology.omega_de_z
.. automethod:: Cosmology.omega_k_z


Densities
---------

.. automethod:: Cosmology.rho_c_z
.. automethod:: Cosmology.rho_m_z
.. automethod:: Cosmology.rho_de_z
.. automethod:: Cosmology.rho_k_z


Distance functions
------------------

.. automethod:: Cosmology.dc
.. automethod:: Cosmology.dm
.. automethod:: Cosmology.da


Dimensionless distance functions
--------------------------------

.. automethod:: Cosmology.xc
.. automethod:: Cosmology.xm
.. automethod:: Cosmology.xa


Inverse distance functions
--------------------------

.. automethod:: Cosmology.dc_inv
.. automethod:: Cosmology.dm_inv
.. automethod:: Cosmology.da_inv
.. automethod:: Cosmology.xc_inv
.. automethod:: Cosmology.xm_inv
.. automethod:: Cosmology.xa_inv


Volume functions
----------------

.. automethod:: Cosmology.dvc
.. automethod:: Cosmology.vc


Creating new adapters
=====================

The :class:`Cosmology` interface is used by subclasses which act as adapters to
concrete implementations of the cosmological background functions.  For example,
the :func:`Cosmology.from_astropy` adapter class starts in the following way::

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

        ...

The :class:`Cosmology` interface fills missing methods in subclasses wherever
possible.  To do so, it tries to implement any missing method in terms of
available methods.  As a result, subclasses mostly only need to provide their
cosmological parameter values and the comoving distance function, and everything
else will be constructed from that.  However, subclasses should implement as
many methods as possible using the wrapped object, or the wrapper might not
return exactly the same values as the wrapped object.


Default implementations
-----------------------

The following table lists all known ways to implement missing methods in terms
of available methods.  Missing methods are filled in from top to bottom.

.. cosmology-default-methods::
