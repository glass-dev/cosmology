# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
************************************
The ``cosmology`` package for Python
************************************

Cosmological background
=======================

.. autosummary::
   :toctree: reference
   :nosignatures:

   LCDM


Large scale structure
=====================

.. autosummary::
   :toctree: reference
   :nosignatures:

   sigma2_r


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

'''

__version__ = '2022.7.12'

__all__ = [
    'LCDM',
    'sigma2_r',
]

from ._lcdm import LCDM
from ._structure import sigma2_r
