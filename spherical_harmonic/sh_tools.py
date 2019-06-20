import numpy as np
import sph_harm as sh


def sh_idx(m, l):
    return l * (l + 1) + m


def print_sh_coeffs(band, coeffs):
    for l in np.arange(band):
        linestr = ""
        for m in np.arange(-l, l+1):
            i = sh_idx(m, l)
            linestr += "{:10.6f}".format(coeffs[i]) + " "
        print(linestr)

        
def spherical_integral(integrand, numsamples):
    dtheta = np.pi / (numsamples - 1)
    dphi = 2*np.pi / (2*numsamples - 1)

    theta = np.arange(0, np.pi, dtheta)
    phi = np.arange(0, np.pi*2, dphi)
    theta, phi = np.meshgrid(theta, phi)
    vals = integrand(theta, phi) * np.sin(theta)

    return vals.sum() * dtheta * dphi


def sh_projection(spherical_func, band, numsamples):
    sh_coeffs = np.ndarray(band**2)
    sh_coeffs.fill(0)

    for l in range(band):
        for m in range(-l, l + 1):
            def integrand(theta, phi):
                return spherical_func(theta, phi) * sh.sph_harm(m, l, theta, phi).real

            i = sh_idx(m, l)
            sh_coeffs[i] = spherical_integral(integrand, numsamples)
    return sh_coeffs


# zonal harmonics
def zh_projection(spherical_func, band, numsamples):
    sh_coeffs = np.ndarray(band**2)
    sh_coeffs.fill(0)

    for l in range(band):
        m = 0

        def integrand(theta, phi):
            return spherical_func(theta, phi) * sh.sph_harm(m, l, theta, phi).real

        i = sh_idx(m, l)
        sh_coeffs[i] = spherical_integral(integrand, numsamples)
    return sh_coeffs


def zh_coeffs_replicate(band, zh_coeffs):
    sh_coeffs = zh_coeffs
    for l in range(band):
        for m in range(-l, l + 1):
            if m != 0:
                base = sh_idx(0, l)
                i = sh_idx(m, l)
                sh_coeffs[i] = zh_coeffs[base]
    return sh_coeffs


def sh_reconstruction(theta, phi, band, sh_coeffs):
    vals = np.zeros_like(theta)
    for l in range(band):
        for m in range(-l, l + 1):
            sh_val = sh.sph_harm(m, l, theta, phi).real
            i = sh_idx(m, l)
            coeff = sh_coeffs[i]
            vals = vals + sh_val * coeff
    return vals