import numpy as np
import spherical as sph
import warp as warp
import BRDF as BRDF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors


class LTC(object):
    """docstring for LTC"""

    def __init__(self, m00=1.0, m02=0.0, m11=1.0, m20=0.0, m22=1.0):
        super(LTC, self).__init__()
        self.m00 = m00
        self.m02 = m02
        self.m11 = m11
        self.m20 = m20
        self.m22 = m22

        self.update_matrix()


    def update_matrix(self):
        self.M = np.array([[self.m00,      0.0, self.m02],
                           [     0.0, self.m11,      0.0],
                           [self.m20,      0.0, self.m22]])
        self.invM = np.linalg.inv(self.M)


    def eval(self, Wi):
        Wi_o_unorm = sph.vec3_transform(self.invM, Wi)

        len = sph.vec3_length(Wi_o_unorm)
        Wi_o = sph.vec3_normalize(Wi_o_unorm)

        # clamped cosine as original function
        D_o = np.clip(sph.vec3_cosTheta(Wi_o), 0, 1) / np.pi

        det = np.linalg.det(self.invM)
        jacobian = det / len**3

        return D_o * jacobian


    def sample(self, sample_):
        Wi_o = warp.square_to_cos_hemisphere(sample_)
        Wi = sph.vec3_transform(self.M, Wi_o)
        Wi = sph.vec3_normalize(Wi)

        is_valid = np.heaviside(sph.vec3_cosTheta(Wi), 0)
        return Wi, is_valid


    def pdf(self, Wi):
        Wi_o_unorm = sph.vec3_transform(self.invM, Wi)

        len = sph.vec3_length(Wi_o_unorm)
        Wi_o = sph.vec3_normalize(Wi_o_unorm)

        pdf_o = warp.square_to_cos_hemisphere_pdf(Wi_o)

        det = np.linalg.det(self.invM)
        jacobian = det / len**3

        return pdf_o * jacobian


if __name__ == "__main__":
    ltc = LTC(m00=0.2, m20=0.5)
    integral = sph.spherical_integral(lambda v: ltc.pdf(v), num_samples=512)
    print(integral)


