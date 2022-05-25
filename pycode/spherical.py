import numpy as np
from NumpyHLSL import float3, Frame


def spherical_dir(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return float3(x, y, z)


def meshgrid_spherical(num_theta_samples, num_phi_samples, hemisphere=False):
    theta = np.linspace(0, np.pi/2 if hemisphere else np.pi, num_theta_samples)
    phi = np.linspace(0, 2 * np.pi, num_phi_samples)
    theta, phi = np.meshgrid(theta, phi)
    return theta, phi


def spherical_integral(integrand, num_samples=128, hemisphere=False):
    theta_max = np.pi
    if hemisphere:
        theta_max = np.pi * 0.5

    phi_max = np.pi * 2

    theta = np.linspace(0, theta_max, num_samples)
    phi = np.linspace(0, phi_max, num_samples)
    theta, phi = np.meshgrid(theta, phi)
    vec = spherical_dir(theta, phi)
    sin_theta = Frame.sinTheta(vec)

    v = integrand(vec)
    assert v.shape == sin_theta.shape, "spherical integration failed: shape mismatch"

    integral = np.sum(v * sin_theta) * theta_max * phi_max / num_samples / num_samples
    return integral


if __name__ == "__main__":
    tolerance = 1e-2

    result = spherical_integral(lambda vec: np.ones((*vec.shape[0:-1], 1)), num_samples=1024) / np.pi
    assert(np.abs(result - 4.0) < tolerance)

    result = spherical_integral(lambda vec: np.clip(Frame.cosTheta(vec), 0.0, 1.0), num_samples=1024) / np.pi
    assert(np.abs(result - 1.0) < tolerance)
