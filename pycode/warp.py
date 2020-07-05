import numpy as np
import spherical as sph


def square_to_cos_hemisphere(sample):
    cosTheta2 = 1 - sample[0]
    cosTheta = np.sqrt(cosTheta2)
    sinTheta = np.sqrt(1 - cosTheta2)
    phi = sample[1] * 2 * np.pi
    return np.array([sinTheta * np.cos(phi), sinTheta * np.sin(phi), cosTheta])


def square_to_cos_hemisphere_pdf(p):
    return np.clip(sph.vec3_cosTheta(p), 0, 1) / np.pi

