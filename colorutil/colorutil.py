import numpy as np
# from line_profiler_pycharm import profile
from numba import jit


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def lrgb_to_srgb(channel):
    srgb_channel = np.where(channel <= 0.0031308, 12.92 * channel, (1.0 + 0.055) * (channel ** (1.0 / 2.4)) - 0.055)
    #     Make sure value is between 0 and 1
    srgb_channel = np.where(srgb_channel > 1, 1, srgb_channel)
    srgb_channel = np.where(srgb_channel < 0, 0, srgb_channel)
    return srgb_channel


# // via https://github.com/tobspr/GLSL-Color-Spaces/blob/master/ColorSpaces.inc.glsl
# const float
# // Converts a single srgb channel to rgb
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def srgb_to_lrgb(channel):
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / (1.0 + 0.055)) ** 2.4)


M_XYZ_TO_LRGB = np.array([
    [3.24096994, -0.96924364, 0.05563008],
    [-1.53738318, 1.8759675, -0.20397696],
    [-0.49861076, 0.04155506, 1.05697151]]
).astype(np.float64)

M_LRGB_TO_XYZ = np.array([
    [0.4123908, 0.21263901, 0.01933082],
    [0.35758434, 0.71516868, 0.11919478],
    [0.18048079, 0.07219232, 0.95053215]]).astype(np.float64)


def lrgb_to_xyz(c):
    return np.dot(c, M_LRGB_TO_XYZ)


@jit(nopython=True, fastmath=True, cache=True)
def xyz_to_lrgb(c):
    e = np.empty(c.shape)
    for i in range(c.shape[0]):
        e[i] = np.dot(c[i], M_XYZ_TO_LRGB)
    return e


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def xyz_to_oklab(c):
    lms = np.empty(c.shape)
    lms[..., 0] = np.cbrt(0.8189330101 * c[..., 0] + 0.3618667424 * c[..., 1] - 0.1288597137 * c[..., 2])
    lms[..., 1] = np.cbrt(0.0329845436 * c[..., 0] + 0.9293118715 * c[..., 1] + 0.0361456387 * c[..., 2])
    lms[..., 2] = np.cbrt(0.0482003018 * c[..., 0] + 0.2643662691 * c[..., 1] + 0.6338517070 * c[..., 2])

    c[..., 0] = 0.2104542553 * lms[..., 0] + 0.7936177850 * lms[..., 1] - 0.0040720468 * lms[..., 2]
    c[..., 0] = np.where(c[..., 0] < 0, 0, c[..., 0])
    c[..., 0] = np.where(c[..., 0] > 1, 1, c[..., 0])
    c[..., 1] = 1.9779984951 * lms[..., 0] - 2.4285922050 * lms[..., 1] + 0.4505937099 * lms[..., 2]
    c[..., 1] = np.where(c[..., 1] < -1, -1, c[..., 1])
    c[..., 1] = np.where(c[..., 1] > 1, 1, c[..., 1])
    c[..., 2] = 0.0259040371 * lms[..., 0] + 0.7827717662 * lms[..., 1] - 0.8086757660 * lms[..., 2]
    c[..., 2] = np.where(c[..., 2] < -1, -1, c[..., 2])
    c[..., 2] = np.where(c[..., 2] > 1, 1, c[..., 2])
    return c


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def oklab_to_xyz(c):
    lms = np.empty(c.shape)
    lms[..., 0] = (c[..., 0] + 0.3963377774 * c[..., 1] + 0.2158037573 * c[..., 2])  # ** 3
    lms[..., 1] = (c[..., 0] - 0.1055613458 * c[..., 1] - 0.0638541728 * c[..., 2])  # ** 3
    lms[..., 2] = (c[..., 0] - 0.0894841775 * c[..., 1] - 1.2914855480 * c[..., 2])  # ** 3

    lms = lms ** 3

    c[..., 0] = 1.2270138511 * lms[..., 0] - 0.5577999807 * lms[..., 1] + 0.2812561490 * lms[..., 2]
    c[..., 1] = -0.0405801784 * lms[..., 0] + 1.1122568696 * lms[..., 1] - 0.0716766787 * lms[..., 2]
    c[..., 2] = -0.0763812845 * lms[..., 0] - 0.4214819784 * lms[..., 1] + 1.5861632204 * lms[..., 2]
    return c


M_XYZD65_TO_XYZD50 = np.array([
    [1.0478112, 0.0228866, -0.0501270],
    [0.0295424, 0.9904844, -0.0170491],
    [-0.0092345, 0.0150436, 0.7521316]
], dtype=np.float64)


def xyzd65_to_xyzd50(c):
    return np.dot(M_XYZD65_TO_XYZD50, c)


# // Adapted from https://observablehq.com/@mbostock/lab-and-rgb
# const vec3 tristimulus_to_xyzd50_1062606552 = (vec3(
# 0.96422, 1, 0.82521
# ));

M_TRISTIMULUS_TO_XYZD50 = np.array([0.96422, 1, 0.82521], dtype=np.float64)


@jit(nopython=True, fastmath=True, cache=True)
def xyzd50_to_lab(c):
    def f_1(t):
        if t > 0.00885645167903563082:
            return t ** (1.0 / 3.0)
        else:
            return 7.78703703703703704 * t + 16.0 / 116.0

    fx = f_1(c[..., 0] / M_TRISTIMULUS_TO_XYZD50[0])
    fy = f_1(c[..., 1] / M_TRISTIMULUS_TO_XYZD50[1])
    fz = f_1(c[..., 2] / M_TRISTIMULUS_TO_XYZD50[2])
    lab = np.empty(c.shape)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


@jit(nopython=True, fastmath=True, cache=True)
def lab_to_xyzd50(c):
    def f_2(t):
        if t > 0.20689655172413793103:
            return t ** 3
        else:
            return (t - 16.0 / 116.0) / 7.78703703703703704

    fy = (c[..., 0] + 16.0) / 116.0
    fx = c[..., 1] / 500.0 + fy
    fz = fy - c[..., 2] / 200.0
    xyzd50 = np.empty(c.shape)
    xyzd50[..., 0] = M_TRISTIMULUS_TO_XYZD50[0] * f_2(fx)
    xyzd50[..., 1] = M_TRISTIMULUS_TO_XYZD50[1] * f_2(fy)
    xyzd50[..., 2] = M_TRISTIMULUS_TO_XYZD50[2] * f_2(fz)
    return xyzd50


def xyz_to_lab(c):
    xyzd50 = xyzd65_to_xyzd50(c)
    lab = xyzd50_to_lab(xyzd50)
    return lab


M_XYZD50_XYZD65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098]
])


def xyzd50_to_xyzd65(c):
    return np.dot(M_XYZD50_XYZD65, c)


M_XYZ_TO_LRGB = np.array([
    [3.24096994, -0.96924364, 0.05563008],
    [-1.53738318, 1.8759675, -0.20397696],
    [-0.49861076, 0.04155506, 1.05697151]
])


def xyz_to_lrgb(c):
    return np.dot(c, M_XYZ_TO_LRGB)


def lab_to_xyz(c):
    xyzd50 = lab_to_xyzd50(c)
    xyzd65 = xyzd50_to_xyzd65(xyzd50)
    return xyzd65


# if __name__ == '__main__':
#     tmp = srgb_to_lrgb(np.array([1, 0, 0]))
#     print(tmp)
#     tmp = lrgb_to_xyz(tmp)
#     print(tmp)
#     tmp = xyz_to_lab(tmp)
#     print(tmp)
#     tmp = lab_to_xyz(tmp)
#     print(tmp)
#     tmp = xyz_to_lrgb(tmp)
#     print(tmp)
#     tmp = lrgb_to_srgb(tmp)
#     print(tmp)
