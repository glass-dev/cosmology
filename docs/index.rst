************************************
The ``cosmology`` package for Python
************************************

The ``cosmology`` package aims to provide a standard interface to the many
different cosmology packages that exist for Python.  For example, it allows you
to start testing your code with e.g. an Astropy cosmology instance, and later
switch to e.g. a CAMB result instance if you happen to require the power spectra
or transfer functions.


.. code:: python

    >>> from cosmology import Cosmology

    >>> # create an Astropy adapter
    >>> from astropy.cosmology import Planck18
    >>> cosmo = Cosmology.from_astropy(Planck18)
    >>> cosmo.omega_m
    0.30966

    >>> # create a CAMB adapter
    >>> import camb
    >>> pars = camb.set_params(H0=70.)
    >>> cosmo = Cosmology.from_camb(pars)
    >>> cosmo.omega_m
    0.2911125273447718


Contents
========

.. toctree::
   :maxdepth: 2

   background


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
