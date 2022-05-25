import numpy as np

# TODO

class Vector3(np.ndarray):
    def __init__(self, a, b, c):
        # if np.isscalar(a):
        #     a = np.array(a)
        # if np.isscalar(b):
        #     b = np.array(b)
        # if np.isscalar(c):
        #     c = np.array(c)
        return np.concatenate(a, b, c)

    @property
    def x(self):
        return self[..., 0:1]

    @x.setter
    def x(self, value):
        self[..., 0:1] = value

    @property
    def y(self):
        return self[..., 1:2]

    @y.setter
    def y(self, value):
        self[..., 1:2] = value

    @property
    def z(self):
        return self[..., 2:3]

    @z.setter
    def z(self, value):
        self[..., 2:3] = value

    @property
    def w(self):
        return self[..., 3:4]

    @w.setter
    def w(self, value):
        self[..., 3:4] = value

    @property
    def r(self):
        return self[..., 0:1]

    @r.setter
    def r(self, value):
        self[..., 0:1] = value

    @property
    def g(self):
        return self[..., 1:2]

    @g.setter
    def g(self, value):
        self[..., 1:2] = value

    @property
    def b(self):
        return self[..., 2:3]

    @b.setter
    def b(self, value):
        self[..., 2:3] = value

    @property
    def a(self):
        return self[..., 3:4]

    @a.setter
    def a(self, value):
        self[..., 3:4] = value

    @property
    def xy(self):
        return self[..., 0:2]

    @xy.setter
    def xy(self, value):
        self[..., 0:2] = value

    @property
    def xyz(self):
        return self[..., 0:3]

    @xyz.setter
    def xyz(self, value):
        self[..., 0:3] = value

    @property
    def rg(self):
        return self[..., 0:2]

    @xy.setter
    def rg(self, value):
        self[..., 0:2] = value

    @property
    def rgb(self):
        return self[..., 0:3]

    @rgb.setter
    def rgb(self, value):
        self[..., 0:3] = value


if __name__ == "__main__":
    v = Vector3(0.0, 1.0, 2.0)
    print(v)
