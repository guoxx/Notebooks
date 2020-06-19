import numpy as np
import spherical as sph
import BRDF as BRDF


class NormalDistributionGGX(object):
    """docstring for NormalDistributionGGX"""
    def __init__(self):
        super(NormalDistributionGGX, self).__init__()


    def eval(self, alpha, NoH):
        return BRDF.NDF_GGX(alpha, NoH)


    def sample(self, alpha, sample_):
        a2 = alpha * alpha

        phi = np.pi * 2 * sample_[0]
        cosTheta = np.sqrt(np.maximum(0, (1 - sample_[1])) / (1 + (a2 - 1) * sample_[1]))
        sinTheta = np.sqrt(np.maximum(0, 1 - cosTheta * cosTheta))

        Wh = np.array([sinTheta * np.cos(phi), sinTheta * np.sin(phi), cosTheta])
        return Wh


    def pdf(self, alpha, Wh):
        clampedCos = np.clip(sph.vec3_cosTheta(Wh), 0, 1)
        return self.eval(alpha, clampedCos) * clampedCos