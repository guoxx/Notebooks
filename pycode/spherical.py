import numpy as np
import enoki as ek

if __name__ == "__main__":
    import mitsuba
    mitsuba.set_variant('packet_rgb')

from mitsuba.core import Float, Vector3f
from mitsuba_ext import Frame


def spherical_dir(theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)
    return Vector3f(x, y, z)


def meshgrid_spherical(num_theta_samples, num_phi_samples, hemisphere=False):
    theta = ek.linspace(Float, 0, np.pi/2 if hemisphere else np.pi, num_theta_samples)
    phi = ek.linspace(Float, 0, 2 * np.pi, num_phi_samples)
    theta, phi = ek.meshgrid(theta, phi)
    return theta, phi


def spherical_integral(integrand, num_samples=128, hemisphere=False):
    theta_max = np.pi
    if hemisphere:
        theta_max = np.pi * 0.5

    phi_max = np.pi * 2

    theta = ek.linspace(Float, 0, theta_max, num_samples)
    phi = ek.linspace(Float, 0, phi_max, num_samples)
    theta, phi = ek.meshgrid(theta, phi)
    vec = spherical_dir(theta, phi)

    v = integrand(vec)
    integral = ek.hsum(v * Frame.sin_theta(vec)) * theta_max * phi_max / num_samples / num_samples
    return integral


if __name__ == "__main__":
    tolerance = 1e-2

    result = spherical_integral(lambda vec : 1, num_samples=1024) / np.pi
    assert(np.abs(result - 4.0) < tolerance)

    result = spherical_integral(lambda vec : ek.clamp(Frame.cos_theta(vec), 0.0, 1.0), num_samples=1024) / np.pi
    assert(np.abs(result - 1.0) < tolerance)
