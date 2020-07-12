if __name__ == "__main__":
    import mitsuba
    mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Float
from mitsuba_ext import Frame

import enoki as ek
import spherical as sph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


MIN_LINEAR_ROUGHNESS = 0.089
MIN_GGX_ALPHA = 0.007921
# MIN_LINEAR_ROUGHNESS = 0.045
# MIN_GGX_ALPHA = 0.002025
FP16_Max = 65504.0
FP16_Min = 0.00006103515625


# Refer to https://google.github.io/filament/Filament.md.html#materialsystem/standardmodel, Section: Roughness remapping and clamping
def clampLinearRoughness(linearRoughness):
    return ek.max(MIN_LINEAR_ROUGHNESS, linearRoughness)


def linearRoughnessToAlpha(linearRoughness):
    return ek.max(MIN_GGX_ALPHA, linearRoughness ** 2)


def NDF_GGX(alpha, NoH):
    a2 = alpha * alpha
    d = ((NoH * a2 - NoH) * NoH + 1)
    return ek.min(a2 * ek.rcp(ek.pi * d * d), FP16_Max)


def NDF_GGX_(alpha, NoH):
    oneMinusSquaredNoH = 1.0 - NoH**2
    a = NoH * alpha
    k = alpha * ek.rcp(a * a + oneMinusSquaredNoH)
    return (k * k * (1.0 / ek.pi))


def fresnelSchlick(f0, f90, cosTheta):
    return f0 + (f90 - f0) * (1 - cosTheta)**5


def smithMaskingFunction(alpha, NdotV):
    denom = 1 + ek.sqrt(1 + alpha*alpha*(1 - NdotV*NdotV)/(NdotV*NdotV))
    return 2.0 * ek.rcp(denom)


def G_GGX(alpha, NdotV, NdotL):
    return smithMaskingFunction(alpha, NdotV) * smithMaskingFunction(alpha, NdotL)


 # Same as G_GGX but have 1/(4*NdotV*NdotL) included
def Gvis_GGX(alpha, NdotV, NdotL):
    a2 = alpha*alpha
    G_V = NdotV + ek.sqrt( (NdotV - NdotV * a2) * NdotV + a2 )
    G_L = NdotL + ek.sqrt( (NdotL - NdotL * a2) * NdotL + a2 )
    return 1 / ( G_V * G_L )


def microfacetBRDF(alpha, specularColor, NdotV, NdotL, NdotH, LdotH):
    D = NDF_GGX(alpha, NdotH)
    Gvis = Gvis_GGX(alpha, NdotV, NdotL)
    F = fresnelSchlick(specularColor, 1, LdotH)
    return F * D * Gvis


def NDF_NormalizationTest(NDF, alpha, num_samples=128):
    def integrand(Wi):
        cosTheta = Frame.cos_theta(Wi)
        return NDF(alpha, cosTheta) * cosTheta

    return sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples) - 1


def maskingFuncProjectedAreaTest(NDF, maskingFunc, alpha, Wo, num_samples=128):
    cosThetaO = Frame.cos_theta(Wo)

    def integrand(Wi):
        cosThetaI = Frame.cos_theta(Wi)
        cosTheta = ek.clamp(ek.dot(Wi, Wo), 0, 1)
        return NDF(alpha, cosThetaI) * maskingFunc(alpha, cosThetaO) * cosTheta

    return sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples) - cosThetaO


def whiteFurnaceTest(NDF, maskingFunc, alpha, Wo, num_samples=128):
    cosThetaO = Frame.cos_theta(Wo)

    def integrand(Wi):
        Wh = ek.normalize(Wi + Wo)
        cosThetaH = Frame.cos_theta(Wh)
        denom = 4 * ek.abs(Frame.cos_theta(Wo))
        mask = (~ek.isnan(cosThetaH)) & (cosThetaH > 0)
        return ek.select(mask, NDF(alpha, cosThetaH) * maskingFunc(alpha, cosThetaO) / denom, 0.0)

    return sph.spherical_integral(integrand, hemisphere=False, num_samples=num_samples) - 1


if __name__ == "__main__":
    alpha = ek.linspace(Float, MIN_GGX_ALPHA, 1, 10)
    err_ndf = ek.zero(Float, len(alpha))
    err_ndf_ = ek.zero(Float, len(alpha))
    for i, a in enumerate(alpha):
        err_ndf[i] = NDF_NormalizationTest(NDF_GGX, a, num_samples=1024)
        err_ndf_[i] = NDF_NormalizationTest(NDF_GGX_, a, num_samples=1024)

    fig = plt.figure(figsize=plt.figaspect(3))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(alpha, np.abs(err_ndf.numpy()), label="NDF_GGX")
    ax1.plot(alpha, np.abs(err_ndf_.numpy()), label="NDF_GGX_")
    ax1.set_ylim(0, 1)
    ax1.legend()

    alpha = ek.linspace(Float, MIN_GGX_ALPHA, 1, 10)
    theta = ek.linspace(Float, 0, ek.pi/2, 10)
    err_proj_area = np.zeros((len(alpha), len(theta)))
    err_furnace_test = np.zeros((len(alpha), len(theta)))
    for i in range(len(alpha)):
        for j in range(len(theta)):
            Wo = sph.spherical_dir(theta[j], 0)
            err_proj_area[i][j] = maskingFuncProjectedAreaTest(NDF_GGX_, smithMaskingFunction, alpha[i], Wo, num_samples=1024)[0]
            err_furnace_test[i][j] = whiteFurnaceTest(NDF_GGX_, smithMaskingFunction, alpha[i], Wo, num_samples=4096)[0]

    alpha = alpha.numpy()
    theta = theta.numpy()
    alpha, theta = np.meshgrid(alpha, theta)

    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.plot_surface(alpha, theta/ek.pi*2, np.abs(err_proj_area))

    ax3 = fig.add_subplot(3, 1, 3, projection='3d')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    ax3.plot_surface(alpha, theta/ek.pi*2, np.abs(err_furnace_test))

    plt.show()
