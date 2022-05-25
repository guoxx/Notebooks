import numpy as np
import matplotlib.pyplot as plt
import spherical as sph
from NumpyHLSL import Frame, normalize, dot, float3


MIN_LINEAR_ROUGHNESS = 0.089
MIN_GGX_ALPHA = 0.007921
# MIN_LINEAR_ROUGHNESS = 0.045
# MIN_GGX_ALPHA = 0.002025
FP16_Max = 65504.0
FP16_Min = 0.00006103515625


# Refer to https://google.github.io/filament/Filament.md.html#materialsystem/standardmodel, Section: Roughness remapping and clamping
def clampLinearRoughness(linearRoughness):
    return np.maximum(MIN_LINEAR_ROUGHNESS, linearRoughness)


def linearRoughnessToAlpha(linearRoughness):
    return np.maximum(MIN_GGX_ALPHA, linearRoughness ** 2)


def NDF_GGX(alpha, NoH):
    a2 = alpha * alpha
    d = ((NoH * a2 - NoH) * NoH + 1)
    return np.minimum(a2 / (np.pi * d * d), FP16_Max)


def NDF_GGX_(alpha, NoH):
    oneMinusSquaredNoH = 1.0 - NoH**2
    a = NoH * alpha
    k = alpha / (a * a + oneMinusSquaredNoH)
    return (k * k * (1.0 / np.pi))


def NDF_GGX_Anisotropic(at, ab, NdotH, TdotH, BdotH):
    a2 = at * ab
    # d = float3(a2 * NdotH, ab * TdotH, at * BdotH)
    # d2 = dot(d, d)
    d2 = (a2 * NdotH)**2 + (ab * TdotH)**2 + (at * BdotH)**2
    b2 = a2 / d2
    return a2 * b2 * b2 * (1.0 / np.pi)


def fresnelSchlick(f0, f90, cosTheta):
    return f0 + (f90 - f0) * (1 - cosTheta)**5


def smithMaskingFunction(alpha, NoV):
    denom = 1 + np.sqrt(1 + alpha*alpha*(1 - NoV*NoV)/(NoV*NoV))
    return 2.0 / denom


def smithAnisotropicMaskingFunction(NdotV, TdotV, BdotV, at, ab):
    NdotV = np.abs(NdotV)
    return 2 * NdotV / (NdotV + np.sqrt((TdotV*at)**2 + (BdotV*ab)**2 + (NdotV)**2))


def G_GGX(alpha, NdotV, NdotL):
    return smithMaskingFunction(alpha, NdotV) * smithMaskingFunction(alpha, NdotL)


def G_GGX_Anisotropic(NdotV, TdotV, BdotV, NdotL, TdotL, BdotL, at, ab):
    return smithAnisotropicMaskingFunction(NdotV, TdotV, BdotV, at, ab) * \
           smithAnisotropicMaskingFunction(NdotL, TdotL,BdotL, at, ab)


 # Same as G_GGX but have 1/(4*NdotV*NdotL) included
def Gvis_GGX(alpha, NoV, NoL):
    a2 = alpha*alpha
    G_V = NoV + np.sqrt( (NoV - NoV * a2) * NoV + a2 )
    G_L = NoL + np.sqrt( (NoL - NoL * a2) * NoL + a2 )
    return 1.0 / ( G_V * G_L )


def microfacetBRDF(alpha, specularColor, NoV, NoL, NoH, LoH):
    D = NDF_GGX(alpha, NoH)
    Gvis = Gvis_GGX(alpha, NoV, NoL)
    F = fresnelSchlick(specularColor, 1, LoH)
    return F * D * Gvis


def NDF_NormalizationTest(NDF, alpha, num_samples=128):
    def integrand(wi):
        cosTheta = Frame.cosTheta(wi)
        return NDF(alpha, cosTheta) * cosTheta

    return sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples) - 1


def maskingFuncProjectedAreaTest(NDF, maskingFunc, alpha, wo, num_samples=128):
    cosThetaO = Frame.cosTheta(wo)

    def integrand(wi):
        cosThetaI = Frame.cosTheta(wi)
        cosTheta = np.clip(dot(wi, wo), 0, 1)
        D = NDF(alpha, cosThetaI)
        G1 = maskingFunc(alpha, cosThetaO)
        v = D * G1 * cosTheta
        return v

    integral = sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples)
    target = cosThetaO
    return integral - target


def whiteFurnaceTest(NDF, maskingFunc, alpha, wo, num_samples=128):
    cosThetaO = Frame.cosTheta(wo)

    def integrand(wi):
        wh = normalize(wi + wo)
        cosThetaH = Frame.cosTheta(wh)
        denom = 4 * np.abs(Frame.cosTheta(wo))
        mask = (~np.isnan(cosThetaH)) & (cosThetaH > 0)
        v = NDF(alpha, cosThetaH) * maskingFunc(alpha, cosThetaO) / denom
        return np.where(mask, v, 0.0)

    integral = sph.spherical_integral(integrand, hemisphere=False, num_samples=num_samples)
    return integral - 1


if __name__ == "__main__":
    alpha = np.linspace(MIN_GGX_ALPHA, 1, 10)
    err_ndf = np.zeros(len(alpha))
    err_ndf_ = np.zeros(len(alpha))
    for i, a in enumerate(alpha):
        err_ndf[i] = NDF_NormalizationTest(NDF_GGX, a, num_samples=1024)
        err_ndf_[i] = NDF_NormalizationTest(NDF_GGX_, a, num_samples=1024)

    fig = plt.figure(figsize=plt.figaspect(3))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(alpha, np.abs(err_ndf), label="NDF_GGX")
    ax1.plot(alpha, np.abs(err_ndf_), label="NDF_GGX_")
    ax1.set_ylim(0, 1)
    ax1.legend()

    alpha = np.linspace(MIN_GGX_ALPHA, 1, 10)
    theta = np.linspace(0, np.pi/2, 10)
    err_proj_area = np.zeros((len(alpha), len(theta)))
    err_furnace_test = np.zeros((len(alpha), len(theta)))
    for i in range(len(alpha)):
        for j in range(len(theta)):
            wo = sph.spherical_dir(theta[j], 0)
            err_proj_area[i][j] = maskingFuncProjectedAreaTest(NDF_GGX_, smithMaskingFunction, alpha[i], wo, num_samples=512)[0]
            err_furnace_test[i][j] = whiteFurnaceTest(NDF_GGX_, smithMaskingFunction, alpha[i], wo, num_samples=1024)

    alpha, theta = np.meshgrid(alpha, theta)

    ax2 = fig.add_subplot(3, 1, 2, projection='3d')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_zlim(0, 1)
    ax2.plot_surface(alpha, theta/np.pi*2, np.abs(err_proj_area))

    ax3 = fig.add_subplot(3, 1, 3, projection='3d')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_zlim(0, 1)
    ax3.plot_surface(alpha, theta/np.pi*2, np.abs(err_furnace_test))

    plt.show()
