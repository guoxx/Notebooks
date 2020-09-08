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
        self.M = Matrix3f(self.m00,      0.0, self.m02,
                               0.0, self.m11,      0.0,
                          self.m20,      0.0, self.m22)
        self.invM = ek.inverse(self.M)


    def eval(self, Wi):
        inv_matrices = self.invM
        ek.set_slices(inv_matrices, ek.slices(Wi))
        Wi_o_unorm = inv_matrices @ Wi

        len = ek.norm(Wi_o_unorm)
        Wi_o = ek.normalize(Wi_o_unorm)

        # clamped cosine as original function
        D_o = ek.clamp(Frame.cos_theta(Wi_o), 0, 1) / np.pi

        det = ek.det(self.invM)
        jacobian = det / len**3

        return D_o * jacobian


    def sample(self, sample_):
        Wi_o = warp.square_to_cosine_hemisphere(sample_)

        matrices = self.M
        ek.set_slices(matrices, ek.slices(Wi_o))
        Wi = matrices @ Wi_o
        Wi = ek.normalize(Wi)

        return Wi, self.pdf(Wi)


    def pdf(self, Wi):
        inv_matrices = self.invM
        ek.set_slices(inv_matrices, ek.slices(Wi))
        Wi_o_unorm = inv_matrices @ Wi

        len = ek.norm(Wi_o_unorm)
        Wi_o = ek.normalize(Wi_o_unorm)

        pdf_o = warp.square_to_cosine_hemisphere_pdf(Wi_o)

        det = ek.det(self.invM)
        jacobian = det / len**3

        return pdf_o * jacobian



class BRDFAdapter(object):
    def __init__(self, roughness, theta_o):
        import BRDF

        self.roughness = BRDF.clampLinearRoughness(roughness)
        self.alpha = BRDF.linearRoughnessToAlpha(self.roughness)
        self.theta_o = theta_o
        self.distr = MicrofacetDistribution(MicrofacetType.GGX, self.alpha)


    def eval(self, Wi):
        Wo = sph.spherical_dir(self.theta_o, 0)
        Wh = ek.normalize(Wo + Wi)
        # D * G / (4 * cosThetaI * cosThetaO) * cosThetaI
        expr = self.distr.eval(Wh) * self.distr.G(Wi, Wo, Wh) / (4 * Frame.cos_theta(Wo))
        return ek.select(Frame.cos_theta(Wi) > 0, expr, 0.0)


    def sample(self, sample_):
        # importance sample GGX distribution
        Wo = sph.spherical_dir(self.theta_o, 0)
        Wh, pdf = self.distr.sample(Wo, sample_)

        Wi = reflect(Wo, Wh)
        dWh_dWi = ek.rcp(4 * ek.dot(Wh, Wi))

        return Wi, ek.select(Frame.cos_theta(Wi) > 0, pdf * dWh_dWi, 0)


    def pdf(self, Wi):
        Wo = sph.spherical_dir(self.theta_o, 0)
        Wh = ek.normalize(Wo + Wi)
        pdf = self.distr.pdf(Wi, Wh)
        dWh_dWi = ek.rcp(4 * ek.dot(Wh, Wi))
        return ek.select(Frame.cos_theta(Wi) > 0, pdf * dWh_dWi, 0)


    def compute_average_term(self):
        # draw uniform distributed samples
        num_samples = 4096
        sample_ = Vector2f(np.random.rand(num_samples, 2))

        # importance sample GGX
        Wi, pdf = self.sample(sample_)
        vals = self.eval(Wi)
        weighted_vals = ek.select(pdf > 0, vals / pdf, 0)

        amplitude = ek.hsum(weighted_vals) / num_samples
        average_dir = Vector3f(ek.hsum(weighted_vals * Wi.x),
                               ek.hsum(weighted_vals * Wi.y),
                               ek.hsum(weighted_vals * Wi.z))
        average_dir = ek.normalize(average_dir)

        average_dir.y = 0
        average_dir = ek.normalize(average_dir)

        return average_dir, amplitude



class FitLTC(object):
    def __init__(self, m00=1.0, m11=1.0, m20=0.0, amplitude=1.0, averageDir=Vector3f(0,0,1)):
        Y = Vector3f(0, 1, 0)
        X = ek.normalize(ek.cross(Y, averageDir))
        worldToLocal = ek.transpose(Matrix3f(X, Y, averageDir))
        localToWorld = ek.inverse(worldToLocal)

        transfo = Matrix3f(m00,   0, 0,
                             0, m11, 0,
                           m20,   0, 1)
        M = localToWorld @ transfo

        self.ltc = LTC(m00=M[0, 0], m02=M[0, 2], m11=M[1, 1], m20=M[2, 0], m22=M[2, 2], amplitude=amplitude)


    # def solve(self, brdf_func):


def plot_ltc(brdf_, ltc_):
    fig = plt.figure(figsize=plt.figaspect(0.33))

    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    num_grid = 50
    theta_, phi = sph.meshgrid_spherical(num_grid, num_grid, hemisphere=False)
    vec = sph.spherical_dir(theta_, phi)

    target_ = brdf_.eval(vec)
    approx_ = ltc_.eval(vec) * ltc_.amplitude

    vmax = ek.max(ek.hmax(target_), ek.hmax(approx_))

    target_ = target_.numpy().reshape(num_grid, num_grid).transpose()
    approx_ = approx_.numpy().reshape(num_grid, num_grid).transpose()
    diff = np.abs(target_ - approx_)

    plot_target = ax0.imshow(target_, cmap='coolwarm', vmax=vmax)
    plot_approx = ax1.imshow(approx_, cmap='coolwarm', vmax=vmax)
    plot_diff = ax2.imshow(diff, cmap='coolwarm', vmax=vmax)
    ax0.title.set_text('BRDF')
    ax1.title.set_text('LTC')
    ax2.title.set_text('Difference')
    fig.colorbar(plot_target, ax=ax0)
    fig.colorbar(plot_approx, ax=ax1)
    fig.colorbar(plot_diff, ax=ax2)

    plt.tight_layout()
    plt.savefig('plots/ltc_roughness-{:.2f}_theta-{:.2f}.png'.format(brdf_.roughness, brdf_.theta_o))
    plt.close(fig)


if __name__ == "__main__":

    ltc_params = np.zeros((8, 16, 5))
    for i, roughness in enumerate(np.linspace(0.2, 1, 8)):
        initial_guess = [1, 1, 1]

        for j, theta in enumerate(np.linspace(0, np.pi/2*0.98, 16)):
            print("roughness {}, theta {} ".format(roughness, theta))

            is_isotropic = (theta == 0)
            initial_guess[2] = 0

            brdf = BRDFAdapter(roughness, theta)

            averageDir, amplitude = brdf.compute_average_term()
            print("average direction ", averageDir)
            print("reflection direction", reflect(sph.spherical_dir(brdf.theta_o, 0), Vector3f(0, 0, 1)))

            num_samples = 4096
            sample_ = Vector2f(np.random.rand(num_samples, 2))

            def compute_error(params, isotropic):
                if isotropic:
                    m00 = params
                    fit = FitLTC(m00=m00, m11=m00, amplitude=amplitude, averageDir=averageDir)
                else:
                    m00, m11, m20 = params
                    fit = FitLTC(m00=m00, m11=m11, m20=m20, amplitude=amplitude, averageDir=averageDir)

                ltc = fit.ltc

                def error(Wi):
                    df = ltc.eval(Wi) * ltc.amplitude - brdf.eval(Wi)
                    return ek.abs(df) ** 3

                # return sph.spherical_integral(error, num_samples=512, hemisphere=False)

                total_error = 0.0
                # num_samples = 4096 * 16

                if True:
                    # sample_ = Vector2f(np.random.rand(num_samples, 2))
                    Wi, pdf_brdf = brdf.sample(sample_)
                    pdf_ltc = ltc.pdf(Wi)
                    sample_weight = ek.select(Frame.cos_theta(Wi) > 0, ek.rcp(pdf_brdf + pdf_ltc), 1000)
                    total_error += ek.hsum(error(Wi) * sample_weight) / num_samples

                if True:
                    # sample_ = Vector2f(np.random.rand(num_samples, 2))
                    Wi, pdf_ltc = ltc.sample(sample_)
                    if ek.any(pdf_ltc < 0):
                        print("negative ltc pdf")
                    pdf_brdf = brdf.pdf(Wi)
                    sample_weight = ek.select(Frame.cos_theta(Wi) > 0, ek.rcp(pdf_brdf + pdf_ltc), 1000)
                    total_error += ek.hsum(error(Wi) * sample_weight) / num_samples

                err0 = np.heaviside(-ltc.m00, 0)
                err1 = np.heaviside(-ltc.m11, 0)
                err2 = np.heaviside(-ltc.m22, 0)
                total_error += (err0 + err1 + err2) * 1000000
                # regularization = ek.abs(ltc.m00-1) + ek.abs(ltc.m02) + ek.abs(ltc.m11-1) + ek.abs(ltc.m20) + ek.abs(ltc.m22 - 1)
                # regularization = regularization[0]
                # regularization = 0
                return total_error

            from scipy import optimize

            if is_isotropic:
                first_guess = [1]
                result = optimize.minimize(lambda params: compute_error(params, is_isotropic), first_guess, method="Nelder-Mead")
                fit = FitLTC(m00=result.x[0], m11=result.x[0], m20=0, amplitude=amplitude, averageDir=averageDir)
                initial_guess = np.array([result.x[0], result.x[0], 0])
            else:
                result = optimize.minimize(lambda params: compute_error(params, is_isotropic), initial_guess, method="Nelder-Mead")
                fit = FitLTC(m00=result.x[0], m11=result.x[1], m20=result.x[2], amplitude=amplitude, averageDir=averageDir)
                initial_guess = np.array([result.x[0], result.x[1], result.x[2]])
            ltc = fit.ltc

            # initial_guess = result.x

            print("result : ", result.x)
            print("ltc normalization ", sph.spherical_integral(ltc.eval, hemisphere=True, num_samples=1024))
            print("ltc pdf integration ", sph.spherical_integral(ltc.pdf, hemisphere=True, num_samples=1024))

            print(np.array([ltc.m00, ltc.m02, ltc.m11, ltc.m20, ltc.m22]).flatten())
            ltc_params[i][j] = np.array([ltc.m00, ltc.m02, ltc.m11, ltc.m20, ltc.m22]).flatten()

            plot_ltc(brdf, ltc)

    # print(ltc_params)