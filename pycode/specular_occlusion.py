import mitsuba
mitsuba.set_variant("packet_rgb")

import enoki as ek

from mitsuba.core import Float, Vector2f, Vector3f
from mitsuba.render import MicrofacetDistribution, MicrofacetType, reflect
from BRDF import linearRoughnessToAlpha

import numpy as np
import matplotlib.pyplot as plt

import spherical as sph

from LTC import BRDFAdapter


def saturate(v):
    return ek.clamp(v, 0, 1)


def specularOcclusion(alphaV, beta, roughness, thetaO):
    thetaC = thetaO - beta
    B = sph.spherical_dir(thetaC, 0)

    numVisSamples = 512
    sample_ = Vector2f(np.random.rand(numVisSamples, 2))

    # negative thetaO
    brdf = BRDFAdapter(roughness, -thetaO)
    Wi, pdf = brdf.sample(sample_)

    visible = ek.select(ek.acos(ek.dot(B, Wi)) < alphaV, Float(1.0), Float(0.0))
    accumValue = ek.select(pdf > 0, visible * brdf.eval(Wi) / pdf, 0)
    accumWeight = ek.select(pdf > 0, brdf.eval(Wi) / pdf, 0)

    accumValue = ek.hsum(accumValue)
    accumWeight = ek.hsum(accumWeight)

    if accumWeight > 0:
        return accumValue / accumWeight
    else:
        return 0


if __name__ == "__main__":
    _SampleCount = 4096
    _ConeHalfAngle = 0.4

    N = Vector3f(0, 0, 1)
    coneDir = N

    # dim = 32
    # lut = np.zeros((32, 16))
    # for i, a  in enumerate(np.linspace(0, 0.1, 32)):
    #     for j, b in enumerate(np.linspace(1, 2.0, 16)):
    #         lut[i][j] = j
    # plt.imshow(lut.transpose(), cmap="Greys")
    # plt.show()

    dim = 32
    lut = np.zeros((dim, dim))
    alphaV = 0.5 * np.pi/2
    beta = 0
    thetaO_ = np.linspace(0, np.pi / 2, dim)
    roughness_ = np.linspace(0.089, 1, dim)
    for i, thetaO in enumerate(thetaO_):
        for j, roughness in enumerate(roughness_):
            lut[i][j] = specularOcclusion(alphaV, beta, roughness, thetaO)

    # This import registers the 3D projection, but is otherwise unused.
    # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X, Y = np.meshgrid(roughness_, thetaO_)
    # surf = ax.plot_surface(X, Y, lut, cmap="coolwarm", linewidth=0, antialiased=False)
    # plt.show()

    plt.imshow(lut, cmap="Greys")
    plt.show()

    # dim = 16
    # lut = np.zeros((dim*dim, dim*dim))
    # for i, alphaV in enumerate(np.linspace(0, np.pi/2, dim)):
    #     for j, beta in enumerate(np.linspace(0, np.pi/2, dim)):
    #         for k, roughness in enumerate(np.linspace(0.089, 1, dim)):
    #             for l, thetaO in enumerate(np.linspace(0, np.pi/2, dim)):
    #                 s = i * dim
    #                 t = j * dim
    #                 lut[s + k][t + l] = specularOcclusion(alphaV, beta, roughness, thetaO)
    #
    # plt.imshow(lut, cmap="Greys")
    # plt.show()

    # dim = 32
    # lut = np.zeros((dim, dim))
    # for i, alphaV in enumerate(np.linspace(0, np.pi / 2, dim)):
    #     for k, roughness in enumerate(np.linspace(0.089, 1, dim)):
    #         for l, thetaO in enumerate(np.linspace(0, np.pi / 2, dim)):
    #             beta = thetaO
    #             lut = specularOcclusion(alphaV, beta, roughness, thetaO)
    #             plt.imshow(lut, cmap="Greys")
    #
    # plt.show()
