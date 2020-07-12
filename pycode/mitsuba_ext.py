import enoki as ek

class Frame:
    def cos_theta(vec):
        return vec.z

    def sin_theta(vec):
        return ek.sqrt(1 - Frame.cos_theta(vec) ** 2)


