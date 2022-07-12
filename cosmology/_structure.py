# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for large scale structure'''

import numpy as np
from scipy.special import loggamma

PI = np.pi
SRPI = PI**0.5


def sigma2_r(k, pk, q=0.0, kr=1.0, window='tophat'):
    r'''mass variance from matter power spectrum

    Computes the mass variance :math:`\sigma(r)` inside a spherical window of
    scale :math:`r` from an input matter power spectrum.  Commonly a tophat
    window is used to produce the variance in spheres of a given radius, but
    other choices are supported.

    The input matter power must be given on a logarithmic grid, and the mass
    variance will returned on a logarithmic grid.  By default, the output grid
    is scaled such that :math:`k_i \, r_{n-i+1} = 1 \forall i = 1, \ldots, n`,
    but can be shifted to other constants using the ``kr`` parameter.

    Parameters
    ----------
    k : array_like (N,)
        Wavenumbers at which the power spectrum is given.  Must have
        logarithmic spacing.
    pk : array_like (..., N)
        Power spectrum for given wavenumbers ``k``.  Can be multidimensional.
        Last axis must agree with the wavenumber axis.
    q : float, optional
        Bias parameter for integral transform.
    kr : float, optional
        Shift parameter for logarithmic output grid.
    window : str, optional
        Type of window function for computing the mass variance.  Supported
        values are ``'tophat'``, ``'gaussian'``.

    Returns
    -------
    r : array_like (N,)
        Scales at which the mass variance is evaluated.
    sigma2_r : array_like (..., N)
        Mass variance in spheres of scale ``r``.  Leading axes correspond to
        the input power spectrum.

    Notes
    -----
    The mass variance is an integral transform of the matter power spectrum
    :math:`P(k)`,

    .. math::

        \sigma^2_r = \frac{1}{2\pi^2}
                        \int_{0}^{\infty} \! P(k) \, k^2 \, w^2(kr) \, dk \;.

    If :math:`P(k)` is given on a logarithmic grid of :math:`k` values, the
    integral can be computed for a logarithmic grid of :math:`r` values with a
    modification of Hamilton's FFTLog algorithm [1]_,

    .. math::

        U(x) = \int_{0}^{\infty} \! t^x \, w^2(t) \, dt \;.

    The implementation supports the usual tophat window function,

    .. math::

        w(x) = \frac{3}{x^3} \, \bigl\{\sin(x) - x \cos(x)\bigr\} \;,

    and the Gaussian window function,

    .. math::

        w(x) = \exp(-x^2/2) \;.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257.
           doi:10.1046/j.1365-8711.2000.03071.x

    '''

    if np.ndim(k) != 1:
        raise TypeError('k must be 1d array')
    if np.shape(pk)[-1] != len(k):
        raise TypeError('last axis of pk must agree with size of k')

    # set up log space k
    lnkr = np.log(kr)
    n = len(k)
    lnk1 = np.log(k[0])
    lnkn = np.log(k[-1])
    lnkc = (lnk1 + lnkn)/2
    dlnk = (lnkn - lnk1)/(n-1)
    jc = (n-1)/2
    j = np.arange(n)

    # make sure given k is linear in log space
    if not np.allclose(k, np.exp(lnkc + (j-jc)*dlnk)):
        raise ValueError('k array not a logarithmic grid')

    # window function
    if window == 'tophat':
        if not -1 < q < 3:
            raise ValueError('bias error: tophap window requires -1 < q < 3')

        def U(x):
            dlg = loggamma((1 + x)/2) - loggamma((4 - x)/2)
            return 9*SRPI*np.exp(dlg)/((4 - x)**2 - 1)

    elif window == 'gaussian':
        if not q > -1:
            raise ValueError('bias error: gaussian window requires q > -1')

        def U(x):
            return np.exp(loggamma((x + 1)/2))/2

    else:
        raise ValueError(f'unknown window function: {window}')

    # low-ringing condition
    y = PI/dlnk
    u = np.exp(-1j*y*lnkr)*U(q + 1j*y)
    a = np.angle(u)/PI
    lnkr = lnkr + dlnk*(a - np.round(a))

    # transform factor
    y = np.linspace(0, 2*PI*(n//2)/(n*dlnk), n//2+1)
    u = np.exp(-1j*y*lnkr)*U(q + 1j*y)

    # ensure that kr is good for n even
    if not n & 1:
        # low ringing kr should make last coefficient real
        if not np.isclose(u[-1].imag, 0):
            raise ValueError('unable to construct low-ringing transform, '
                             'try odd number of points or different q')

        # fix last coefficient to be real
        u.imag[-1] = 0

    # transform via real FFT
    xi = np.fft.rfft(pk*k**(2-q), axis=-1)
    xi *= u
    xi = np.fft.irfft(xi, n, axis=-1)
    xi[..., :] = xi[..., ::-1]

    # set up r in log space
    r = np.exp(lnkr)/k[::-1]

    # prefactor for output
    xi /= 2*PI**2
    xi /= r**(1+q)

    # done, return separations and correlations
    return r, xi
