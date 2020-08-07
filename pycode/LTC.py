if __name__ == "__main__":
    import mitsuba
    mitsuba.set_variant('packet_rgb')

import numpy as np
import enoki as ek

from mitsuba.core import Float, Vector2f, Vector3f, Matrix3f, Matrix4f, Transform4f, warp
from mitsuba.render import MicrofacetDistribution, MicrofacetType, reflect
from mitsuba_ext import Frame

import matplotlib.pyplot as plt
import plot as pltx

import spherical as sph


class LTC(object):
    """docstring for LTC"""

    def __init__(self, m00=1.0, m02=0.0, m11=1.0, m20=0.0, m22=1.0, amplitude=1.0):
        super(LTC, self).__init__()
        self.m00 = m00
        self.m02 = m02
        self.m11 = m11
        self.m20 = m20
        self.m22 = m22
        self.amplitude = amplitude

        self.update_matrix()


    def update_matrix(self):
        self.M = Matrix4f(self.m00,      0.0, self.m02, 0,
                               0.0, self.m11,      0.0, 0,
                          self.m20,      0.0, self.m22, 0,
                                 0,        0,        0, 1)
        transfo = Transform4f(self.M)
        self.invM = transfo.inverse().matrix

        test = self.M @ self.invM
        aaa = 0


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


class FitLTC(object):
    def __init__(self, m00=1.0, m11=1.0, m20=0.0, amplitude=1.0, averageDir=Vector3f(0,0,1)):
        Y = Vector3f(0, 1, 0)
        X = ek.normalize(ek.cross(Y, averageDir))
        worldToLocal = Matrix3f(X, Y, averageDir)
        localToWorld = ek.inverse(worldToLocal)
        transfo = Matrix3f(m00,   0, 0,
                             0, m11, 0,
                           m20,   0, 1)
        # M = transfo @ basis
        M = localToWorld @ transfo
        print(M)

        self.ltc = LTC(m00=M[0][0], m02=M[0][2], m11=M[1][1], m20=M[2][0], m22=M[2][2], amplitude=amplitude)
        # self.ltc = LTC()


if __name__ == "__main__":

    alpha = 0.25
    theta = np.pi/2*0.4

    ggx = MicrofacetDistribution(MicrofacetType.GGX, alpha)

    def BRDF(Wi):
        Wo = sph.spherical_dir(theta, 0)
        Wh = ek.normalize(Wo + Wi)
        return ggx.eval(Wh) * ggx.G(Wi, Wo, Wh) / (4 * Frame.cos_theta(Wo))

    # integral = sph.spherical_integral(lambda Wi : BRDF(Wi), hemisphere=True, num_samples=1024)

    averageDir = Vector3f(0)
    amplitude = 0

    sample_cnt = 4096*16
    for i in range(sample_cnt):
        Wo = sph.spherical_dir(theta, 0)

        # draw uniform distributed samples
        sample_ = np.random.rand(2)
        sample_ = Vector2f(sample_[0], sample_[1])

        # importance sample GGX distribution
        Wh, pdf = ggx.sample(Wo, sample_)

        Wi = reflect(Wo, Wh)
        dWh_dWi = ek.rcp(4 * ek.dot(Wh, Wi))
        sample_weight = ek.rcp(pdf * dWh_dWi) / sample_cnt

        v = BRDF(Wi) * sample_weight
        amplitude += v
        averageDir += v * Wi

    averageDir.y = 0
    averageDir = ek.normalize(averageDir)
    print("average direction ", averageDir)
    print("reflection direction", reflect(sph.spherical_dir(theta, 0), Vector3f(0, 0, 1)))


    def compute_error_isotropic(params):
        m00, m11 = params
        fit = FitLTC(m00=m00, m11=m11, amplitude=amplitude, averageDir=averageDir)
        ltc = fit.ltc
        # ltc = LTC(m00=m00, m11=m11, amplitude=amplitude)

        def diff(Wi):
            df = ltc.eval(Wi) * ltc.amplitude - BRDF(Wi)
            return ek.abs(df) ** 3
        return sph.spherical_integral(diff, num_samples=256, hemisphere=True)

    def compute_error(params):
        m00, m11, m20 = params
        fit = FitLTC(m00=m00, m11=m11, m20=m20, amplitude=amplitude, averageDir=averageDir)
        ltc = fit.ltc
        # ltc = LTC(m00=m00, m11=m11, amplitude=amplitude)

        def diff(Wi):
            df = ltc.eval(Wi) * ltc.amplitude - BRDF(Wi)
            return ek.abs(df) ** 3
        return sph.spherical_integral(diff, num_samples=256, hemisphere=True)

    from scipy import optimize

    initial_guess = [1, 1, 0]
    result = optimize.minimize(compute_error, initial_guess, method="Nelder-Mead")
    # ltc = LTC(m00=result.x[0], m11=result.x[1], amplitude=amplitude)
    fit = FitLTC(m00=result.x[0], m11=result.x[1], m20=result.x[2], amplitude=amplitude, averageDir=averageDir)
    ltc = fit.ltc
    print(result)

    print("ltc normalization ", sph.spherical_integral(ltc.eval, hemisphere=True, num_samples=1024))
    print("ltc pdf integration ", sph.spherical_integral(ltc.pdf, hemisphere=True, num_samples=1024))

    fig = plt.figure(figsize=plt.figaspect(2))
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    pltx.spherical_plot2d(BRDF, axes=ax0)
    pltx.spherical_plot2d(lambda Wi: ltc.eval(Wi) * ltc.amplitude, axes=ax1)
    plt.show()

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

    chi2.run()
    chi2._dump_tables()


