import numpy as np
import matplotlib.pyplot as plt

from pycode.BRDF import linearRoughnessToAlpha
from pycode.NumpyHLSL import frac, Frame, dot, float3
from pycode.MicrofacetDistribution import MicrofacetReflection, BSDFSamplingRecord


def nthSampleR2Sequence(n):
    magic = np.array([0.7548776662466927, 0.5698402909980532])
    return frac(0.5 + magic * (n + 1)).astype(np.float32)


def specularOcclusion(alphaV, cosBeta, roughness):
    beta = np.arccos(cosBeta)

    thetaC_0 = -beta / 2
    thetaO_0 = beta / 2

    thetaO_1 = np.minimum(beta, np.pi / 2)
    thetaC_1 = thetaO_1 - beta

    # 0.78 is a magic number that manually tweak for our reference implementation
    factor = 0.78
    thetaO = thetaO_0 * (1 - factor) + thetaO_1 * factor
    thetaC = thetaC_0 * (1 - factor) + thetaC_1 * factor

    B = float3(np.sin(thetaC), np.zeros_like(thetaC), np.cos(thetaC))
    Nx = float3(0, 0, 1)
    Wr = float3(np.sin(thetaO), np.zeros_like(thetaO), np.cos(thetaO))

    frame = Frame(Nx, T=np.where(dot(Nx, Wr) >= 0.99, float3(0, 1, 0), Wr))

    V = frame.toWorld(Wr)

    sampleCount = 256

    accumValue = 0
    accumWeight = 0

    for i in range(sampleCount):
        u_ = nthSampleR2Sequence(i)

        ggxAlpha = linearRoughnessToAlpha(roughness)

        bRec = BSDFSamplingRecord(frame.toLocal(V))
        bsdf = MicrofacetReflection(ggxAlpha, ggxAlpha)

        sampleWeight = bsdf.sample(bRec, u_)
        sampleWeight *= Frame.cosTheta(bRec.wi)

        wi = frame.toWorld(bRec.wi)
        cosThetaI = Frame.cosTheta(bRec.wi)

        sampleWeight = np.where(cosThetaI > 0, sampleWeight, 0)
        visible = np.where(np.arccos(dot(B, wi)) < np.expand_dims(alphaV, axis=-1) * np.pi * 0.5, 1.0, 0.0)
        accumValue += visible * sampleWeight
        accumWeight += sampleWeight

    return np.where(accumWeight > 0, accumValue / accumWeight, 0)


if __name__ == "__main__":
    dim = 64
    lut = np.zeros((dim, dim))

    roughness = np.linspace(0.089, 1, dim)
    cosBeta = np.cos(np.linspace(0, np.pi, dim))
    roughness, cosBeta = np.meshgrid(roughness, cosBeta)
    alphaV = np.ones_like(cosBeta) * 0.25
    lut = specularOcclusion(alphaV, cosBeta, roughness)

    plt.imshow(lut, cmap="gray")
    plt.show()



