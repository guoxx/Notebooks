import numpy as np
from VectorMath import Vector


def saturate(v):
    return np.clip(v, 0, 1)


def frac(v):
    return np.modf(v)[0]


def abs(v):
    return np.absolute(v)

def max(a, b):
    return np.maximum(a, b)

def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape[-1] == 3 and b.shape[-1] == 3
    a_v = a.view(Vector)
    b_v = b.view(Vector)
    return a_v.x * b_v.x + a_v.y * b_v.y + a_v.z * b_v.z


def normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=-1), -1)
    return v/norm


def rsqrt(v):
    return 1.0/np.sqrt(v)


def utReflect(v, n):
    return -v + 2 * dot(v, n) * n


def float3(v0, v1, v2, keepdims=True):
    if keepdims:
        return np.stack((v0, v1, v2), axis=-1)
    else:
        return np.concatenate((v0, v1, v2), axis=-1)


class Frame:
    def __init__(self, N, *, T=None):
        self.normal = N

        if T is not None:
            T_ = T - N * dot(T, N)
            self.tangent = normalize(T_)
        else:
            self.tangent = Frame.__getPerpendicular(N)

        self.bitangent = np.cross(self.normal, self.tangent)

    def toLocal(self, vec):
        v_local = float3(dot(vec, self.tangent), dot(vec, self.bitangent), dot(vec, self.normal), keepdims=False)
        assert v_local.shape == vec.shape
        return v_local

    def toWorld(self, vec):
        vec_v = vec.view(Vector)
        return vec_v.x * self.tangent + vec_v.y * self.bitangent + vec_v.z * self.normal

    def cosTheta(v):
        return v.view(Vector).z

    def sinTheta(v):
        return np.sqrt(1 - Frame.cosTheta(v) ** 2)

    def __getPerpendicular(a):
        a_v = a.view(Vector)
        xz_or_yz = abs(a_v.x) > abs(a_v.y)
        len = np.where(xz_or_yz,
                       np.sqrt(a_v.x * a_v.x + a_v.z * a_v.z),
                       np.sqrt(a_v.y * a_v.y + a_v.z * a_v.z))
        invLen = 1.0 / len

        return np.where(xz_or_yz,
                        float3(a_v.z * invLen, np.zeros_like(a_v.z), -a_v.x * invLen),
                        float3(np.zeros_like(a_v.z), a_v.z * invLen, -a_v.y * invLen))
