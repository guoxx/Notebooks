import numpy as np
import BRDF as BRDF
import spherical as sph
from NumpyHLSL import normalize, rsqrt, utReflect, float3, Frame, dot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class BSDFSamplingRecord:
    def __init__(self, wo, *, wi=None):
        self.wo = wo
        if wi is not None:
            self.wi = wi

        self.eta = 1


class MicrofacetDistribution:
    def __init__(self, alpha_x, alpha_y):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

    def effectivelySmooth(self):
        return np.maximum(self.alpha_x, self.alpha_y) < 1e-3

    def D(self, wh):
        return BRDF.NDF_GGX_Anisotropic(self.alpha_x, self.alpha_y, wh[..., 2:3], wh[..., 0:1], wh[..., 1:2])

    def G1(self, w):
        return BRDF.smithAnisotropicMaskingFunction(w[..., 2:3], w[..., 0:1], w[..., 1:2], self.alpha_x, self.alpha_y)

    def G(self, wo, wi):
        return BRDF.G_GGX_Anisotropic(wo[..., 2:3], wo[..., 0:1], wo[..., 1:2], wi[..., 2:3], wi[..., 0:1], wi[..., 1:2], self.alpha_x, self.alpha_y)

    # Refer to http://jcgt.org/published/0007/04/01/
    def sample_wh(self, wo, u):
        U1 = u[..., 0:1]
        U2 = u[..., 1:2]
        Ve = wo

        foo = self.alpha_x * Ve[..., 0]

        # Section 3.2: transforming the view direction to the hemisphere configuration
        Vh = normalize(float3(self.alpha_x * Ve[..., 0], self.alpha_y * Ve[..., 1], Ve[..., 2]))
        # Section 4.1: orthonormal basis (with special case if cross product is zero)
        lensq = Vh[..., 0] * Vh[..., 0] + Vh[..., 1] * Vh[..., 1]
        T1 = float3(-Vh[..., 1], Vh[..., 0], np.zeros_like(Vh[..., 0])) * rsqrt(lensq) if lensq > 0 else float3(1, 0, 0)
        T2 = np.cross(Vh, T1)
        # Section 4.2: parameterization of the projected area
        r = np.sqrt(U1)
        phi = 2.0 * np.pi * U2
        t1 = r * np.cos(phi)
        t2 = r * np.sin(phi)
        s = 0.5 * (1.0 + Vh[..., 2:3])
        t2 = (1.0 - s) * np.sqrt(1.0 - t1 * t1) + s * t2
        # Section 4.3: reprojection onto hemisphere
        Nh = t1 * T1 + t2 * T2 + np.sqrt(np.maximum(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh
        # Section 3.4: transforming the normal back to the ellipsoid configuration
        Ne = normalize(float3(self.alpha_x * Nh[..., 0], self.alpha_y * Nh[..., 1], np.maximum(0.0, Nh[..., 2])))
        return Ne

    def pdf(self, wo, wh):
        return self.D(wh) * self.G1(wo) * np.abs(np.linalg.dot(wo, wh)) / np.abs(Frame.cosTheta(wo))


class MicrofacetReflection:
    def __init__(self, alpha_x, alpha_y):
        self.distr = MicrofacetDistribution(alpha_x, alpha_y)

    def sample(self, bRec, sample_):
        wh = self.distr.sample_wh(bRec.wo, sample_)
        wh *= np.sign(bRec.wo[..., 2:3])

        wi = utReflect(bRec.wo, wh)

        is_valid = Frame.cosTheta(bRec.wo) * Frame.cosTheta(wi) > 0

        bRec.wi = np.where(is_valid, wi, np.array([0,0,0]))

        cosThetaO = Frame.cosTheta(bRec.wo)
        cosThetaI = Frame.cosTheta(bRec.wi)
        pdf_ = self.distr.G(cosThetaO, cosThetaI) / np.abs(self.distr.G1(cosThetaO) * cosThetaI)
        return np.where(is_valid, pdf_, 0)

    def eval(self, bRec):
        is_valid = Frame.cosTheta(bRec.wo) * Frame.cosTheta(bRec.wi) > 0
        wh = normalize(bRec.wo + bRec.wi)
        f = self.distr.D(wh) * self.distr.G(bRec.wo, bRec.wi) / (4 * (Frame.cosTheta(bRec.wo) * Frame.cosTheta(bRec.wi)))
        return np.where(is_valid, f, 0)

    def pdf(self, bRec):
        is_valid = Frame.cosTheta(bRec.wo) * Frame.cosTheta(bRec.wi) > 0

        wh = normalize(bRec.wo + bRec.wi)
        dwh_dwi = 1.0 / (4.0 * np.abs(dot(bRec.wi, wh)))
        return np.where(is_valid, self.distr.pdf(bRec.wo, wh) * dwh_dwi, 0)


if __name__ == "__main__":
    refl = MicrofacetReflection(0.8, 0.8)
    wo = sph.spherical_dir(70.0 / 180 * np.pi, 0)

    theta_i, phi_i = sph.meshgrid_spherical(256, 256, True)
    wi = sph.spherical_dir(theta_i, phi_i)

    bRec = BSDFSamplingRecord(wo)
    refl.sample(bRec, np.array([0.2,0.3]))

    bRec = BSDFSamplingRecord(wo, wi=wi)
    bsdf = refl.eval(bRec)
    v = bsdf * Frame.cosTheta(wi)
    v = v.reshape(v.shape[0:-1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(wi[..., 0] * v, wi[..., 1] * v, wi[..., 2] * v, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    # ax.set_lim(-1.0, 1.0)
    # ax.set_lim(-1.0, 1.0)
    # # Customize the z axis.
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

