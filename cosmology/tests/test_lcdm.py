import pytest
import numpy as np
from cosmology import LCDM
from scipy.special import hyp2f1


def test_xc_flat():
    h = np.random.rand()
    Om = np.random.rand()

    cosmo = LCDM(H0=100*h, Om=Om)

    z = np.linspace(0., cosmo.zmax-1, int(cosmo.zmax))
    xc = cosmo.xc(z)

    h1 = hyp2f1(1/3, 1/2, 4/3, -Om/(1 - Om)*(1 + z)**3)
    h2 = hyp2f1(1/3, 1/2, 4/3, -Om/(1 - Om))
    exact = ((1 + z)*h1 - h2)/(1 - Om)**0.5

    np.testing.assert_allclose(xc, exact, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize('k', [0, +1, -1])
def test_xc_inv(k):
    h = np.random.rand()
    Om = np.random.rand()
    Ok = k*np.random.rand()

    cosmo = LCDM(H0=100*h, Om=Om, Ok=Ok)

    z = np.linspace(0., cosmo.zmax-1, int(cosmo.zmax))
    xc = cosmo.xc(z)

    xc_inv = cosmo.xc_inv(xc)

    np.testing.assert_allclose(xc_inv, z)
