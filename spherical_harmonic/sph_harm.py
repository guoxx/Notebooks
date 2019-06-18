import numpy as np
from scipy.misc import factorial


def P(l, m, x):
    # evaluate an Associated Legendre Polynomial P(l,m,x) at x
    pmm = 1.0;
    if (m > 0):
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        i = 1
        while i <= m:
            pmm *= (-fact) * somx2
            fact += 2.0
            i += 1

    if (l == m):
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if (l == m + 1):
        return pmmp1

    pll = 0.0
    ll = m + 2
    while ll <= l:
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
        ll += 1

    return pll


def K(l, m):
    #  renormalisation constant for SH function
    temp = ((2.0 * l + 1.0) * factorial(l - m)) / (4.0 * np.pi * factorial(l + m))
    return np.sqrt(temp)


def sph_harm(m, l, theta, phi):
    #  return a point sample of a Spherical Harmonic basis function
    #  l is the band, range [0..N]
    #  m in the range [-l..l]
    #  theta in the range [0..Pi]
    #  phi in the range [0..2*Pi]
    sqrt2 = np.sqrt(2.0)
    if (m == 0):
        return K(l, 0) * P(l, m, np.cos(theta))
    elif (m > 0):
        return sqrt2 * K(l, m) * np.cos(m * phi) * P(l, m, np.cos(theta))
    else:
        return sqrt2 * K(l, -m) * np.sin(-m * phi) * P(l, -m, np.cos(theta))
    
    
def sh_idx(m, l):
    return l * (l + 1) + m