import numpy as np
import BRDF as BRDF
import spherical as sph


class MicrofacetReflection(object):
    """docstring for MicrofacetReflection"""

    def __init__(self, linearRoughness):
        super(MicrofacetReflection, self).__init__()
        self.linearRoughness = BRDF.clampLinearRoughness(linearRoughness)
        self.alpha = BRDF.linearRoughnessToAlpha(self.linearRoughness)


    def eavl(self, Wo, Wi):
        N = np.array([0, 0, 1])
        Wh = sph.vec3_normalize(Wo + Wi)

        NoH = np.clip(sph.vec3_normalize(Wh), 0, 1)
        NoV = np.clip(sph.vec3_dot(N, Wo), 0, 1)
        NoL = np.clip(sph.vec3_dot(N, Wi), 0, 1)

        D = BRDF.NDF_GGX(self.alpha, NoH)
        Gvis = BRDF.Gvis_GGX(self.alpha, NoV, NoL)
        return D * Gvis


