if __name__ == "__main__":
    import mitsuba
    mitsuba.set_variant('packet_rgb')

import numpy as np
import enoki as ek

from mitsuba.core import Float, Matrix4f, Transform4f, warp
from mitsuba_ext import Frame

import spherical as sph


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
        self.M = Matrix4f(self.m00,      0.0, self.m02, 0,
                               0.0, self.m11,      0.0, 0,
                          self.m20,      0.0, self.m22, 0,
                                 0,        0,        0, 1)
        transfo = Transform4f(self.M)
        self.invM = transfo.inverse().matrix


    def eval(self, Wi):
        inv_transfo = Transform4f(self.invM)
        Wi_o_unorm = inv_transfo.transform_vector(Wi)

        len = ek.norm(Wi_o_unorm)
        Wi_o = ek.normalize(Wi_o_unorm)

        # clamped cosine as original function
        D_o = ek.clamp(Frame.cos_theta(Wi_o), 0, 1) / np.pi

        det = ek.det(self.invM)
        jacobian = det / len**3

        return D_o * jacobian


    def sample(self, sample_):
        Wi_o = warp.square_to_cosine_hemisphere(sample_)

        transfo = Transform4f(self.M)
        Wi = transfo.transform_vector(Wi_o)
        Wi = ek.normalize(Wi)

        return Wi, ek.select(Frame.cos_theta(Wi) > 0, Float(1.0), Float(0.0))


    def pdf(self, Wi):
        inv_transfo = Transform4f(self.invM)
        Wi_o_unorm = inv_transfo.transform_vector(Wi)

        len = ek.norm(Wi_o_unorm)
        Wi_o = ek.normalize(Wi_o_unorm)

        pdf_o = warp.square_to_cosine_hemisphere_pdf(Wi_o)

        det = ek.det(self.invM)
        jacobian = det / len**3

        return pdf_o * jacobian


if __name__ == "__main__":
    ltc = LTC(0.2, 0.2, 1)


    from mitsuba.python.chi2 import ChiSquareTest, SphericalDomain

    # some sampling code
    def my_sample(sample):
        return ltc.sample(sample)

    # the corresponding probability density function
    def my_pdf(p):
        return ltc.pdf(p)

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=my_sample,
        pdf_func=my_pdf,
        sample_dim=2
    )

    assert chi2.run()


