import numpy as np


def saturate(v):
    return np.clip(v, 0, 1)


def frac(v):
    return np.modf(v)[0]


def abs(v):
    return np.absolute(v)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape[-1] == 3 and b.shape[-1] == 3
    return a[..., 0:1] * b[..., 0:1] + a[..., 1:2] * b[..., 1:2] + a[..., 2:3] * b[..., 2:3]


def normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=-1), -1)
    return v/norm


def rsqrt(v):
    return 1.0/np.sqrt(v)


def utReflect(v, n):
    return -v + 2 * dot(v, n) * n


def float3(v0, v1, v2):
    return np.stack((v0, v1, v2), axis=-1)


class Frame:
    def __init__(self, N, *, T=None):
        self.normal = N

        if T is not None:
            T_ = T - N * dot(T, N)
            self.tangent = T_ / np.linalg.norm(T_)
        else:
            self.tangent = Frame.__getPerpendicular(N)

        self.bitangent = np.cross(self.normal, self.tangent)

    def toLocal(self, v):
        return float3(dot(v, self.tangent), dot(v, self.bitangent), dot(v, self.normal))

    def toWorld(self, v):
        return v[..., 0:1] * self.tangent + v[..., 1:2] * self.bitangent + v[..., 2:3] * self.normal

    def cosTheta(v):
        return v[..., 2:3]

    def sinTheta(v):
        return np.sqrt(1 - Frame.cosTheta(v) ** 2)

    def __getPerpendicular(a):
        if (abs(a[..., 0:1]) > abs(a[..., 1:2])):
            invLen = 1.0 / np.sqrt(a[..., 0:1] * a[..., 0:1] + a[..., 2:3] * a[..., 2:3])
            c = float3(a[..., 2:3] * invLen, np.zeros_like(a)[..., 0:1], -a[..., 0:1] * invLen)
        else:
            invLen = 1.0 / np.sqrt(a[..., 1:2] * a[..., 1:2] + a[..., 2:3] * a[..., 2:3])
            c = float3(np.zeros_like(a)[..., 0:1], a[..., 2:3] * invLen, -a[..., 1:2] * invLen)
        return c
