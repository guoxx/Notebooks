import ipyvolume as ipv
import spherical as sph


def spherical_plot3d(func, color=[1., 0, 0, 1.], num_samples=128, clear=True):
    theta, phi = sph.meshgrid_spherical(num_samples, num_samples)
    vec = sph.spherical_dir(theta, phi)
    vals = func(vec)
    points = vals * vec

    if clear:
        ipv.clear()

    px = points.x.numpy().reshape(num_samples, num_samples)
    py = points.y.numpy().reshape(num_samples, num_samples)
    pz = points.z.numpy().reshape(num_samples, num_samples)

    if color[3] < 1.0:
        obj = ipv.plot_mesh(px, pz, py, wireframe=False, color=color)
        obj.material.transparent = True
        obj.material.side = "FrontSide"
    else:
        obj = ipv.plot_mesh(px, pz, py, wireframe=False, color=color)

    return obj