import numpy as np
import ipyvolume as ipv


def spherical_dir(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def spherical_coord(vec):
    x, y, z = vec[0], vec[1], vec[2]
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / norm)
    phi = np.arctan2(y, x)
    return theta, phi


def vec3_cosTheta(vec):
    return vec[2]


def vec3_sinTheta(vec):
    return np.sqrt(1 - vec[2]**2)


def vec3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_length(vec):
    return np.sqrt(vec3_dot(vec, vec))


def vec3_normalize(vec):
    l = vec3_length(vec)
    return np.array([vec[0] / l, vec[1] / l, vec[2] / l])


def vec3_transform(vec, m):
    x, y, z = vec[0], vec[1], vec[2]
    x_ = x * m[0][0] + y * m[0][1] + z * m[0][2]
    y_ = x * m[1][0] + y * m[1][1] + z * m[1][2]
    z_ = x * m[2][0] + y * m[2][1] + z * m[2][2]
    return np.array(x_, y_, z_)


def meshgrid_spherical_coord(num_samples):
    theta = np.linspace(0, np.pi, num=num_samples)
    phi = np.linspace(0, 2 * np.pi, num=num_samples * 2)
    theta, phi = np.meshgrid(theta, phi)
    return theta, phi


def spherical_plot3d(func, color=[1., 0, 0, 1.], num_samples=128, clear=True):
    theta, phi = meshgrid_spherical_coord(num_samples)
    vec = spherical_dir(theta, phi)
    vals = func(vec)
    points = vals * vec

    if clear:
        ipv.clear()
    
    if color[3] < 1.0:
        obj = ipv.plot_mesh(points[0], points[2], points[1], wireframe=False, color=color)
        obj.material.transparent = True
        obj.material.side = "FrontSide"
    else:
        obj = ipv.plot_mesh(points[0], points[2], points[1], wireframe=False, color=color)

    return obj


def spherical_integral(integrand, num_samples=128, hemisphere=False):
    theta_max = np.pi
    if hemisphere:
        theta_max = np.pi * 0.5

    phi_max = np.pi * 2

    theta = np.linspace(0, theta_max, num_samples)
    phi = np.linspace(0, phi_max, num_samples)
    theta, phi = np.meshgrid(theta, phi)
    vec = spherical_dir(theta, phi)

    v = integrand(vec)
    integral = np.sum(integrand(vec) * vec3_sinTheta(vec)) * theta_max * phi_max / num_samples / num_samples
    return integral
