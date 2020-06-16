import numpy as np
import spherical as sph


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
    oneMinusSquaredNoH = 1 - NoH**2
    a = NoH * alpha
    k = alpha * (1 / (a * a + oneMinusSquaredNoH))
    return (k * k * (1 / np.pi))


def fresnelSchlick(f0, f90, cosTheta):
    return f0 + (f90 - f0) * (1 - cosTheta)**5


def smithMaskingFunction(alpha, NdotV):
    denom = 1 + np.sqrt(1 + alpha*alpha*(1 - NdotV*NdotV)/(NdotV*NdotV))
    return 2.0 / denom


def G_GGX(alpha, NdotV, NdotL):
    return smithMaskingFunction(alpha, NdotV) * smithMaskingFunction(alpha, NdotL)


 # Same as G_GGX but have 1/(4*NdotV*NdotL) included
def Gvis_GGX(alpha, NdotV, NdotL):
    a2 = alpha*alpha
    G_V = NdotV + np.sqrt( (NdotV - NdotV * a2) * NdotV + a2 )
    G_L = NdotL + np.sqrt( (NdotL - NdotL * a2) * NdotL + a2 )
    return 1 / ( G_V * G_L )


def microfacetBRDF(alpha, specularColor, NdotV, NdotL, NdotH, LdotH):
    D = NDF_GGX(alpha, NdotH)
    Gvis = Gvis_GGX(alpha, NdotV, NdotL)
    F = fresnelSchlick(specularColor, 1, LdotH)
    return F * D * Gvis


def NDF_NormalizationTest(NDF, alpha, num_samples=128):
    def integrand(Wi):
        cosTheta = sph.vec3_cosTheta(Wi)
        return NDF(alpha, cosTheta) * cosTheta

    return sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples) - 1


def maskingFuncProjectedAreaTest(NDF, maskingFunc, alpha, Wo, num_samples=128):
    cosThetaO = sph.vec3_cosTheta(Wo)

    def integrand(Wi):
        cosThetaI = sph.vec3_cosTheta(Wi)
        cosTheta = np.clip(sph.vec3_dot(Wi, Wo), 0, 1)
        return NDF(alpha, cosThetaI) * maskingFunc(alpha, cosThetaO) * cosTheta

    return sph.spherical_integral(integrand, hemisphere=True, num_samples=num_samples) - cosThetaO