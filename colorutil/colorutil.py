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
#
M_TRISTIMULUS_TO_XYZD50 = np.array([0.96422, 1, 0.82521], dtype=np.float64)


@jit(nopython=True, fastmath=True, cache=True)
def xyzd50_to_lab(c):
    def f_1(t):
        return np.where(t > 0.00885645167903563082, t ** (1.0 / 3.0), 7.78703703703703704 * t + 16.0 / 116.0)

    fx = f_1(c[..., 0] / M_TRISTIMULUS_TO_XYZD50[0])
    fy = f_1(c[..., 1] / M_TRISTIMULUS_TO_XYZD50[1])
    fz = f_1(c[..., 2] / M_TRISTIMULUS_TO_XYZD50[2])
    lab = np.empty(c.shape)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


# 0.95047 1.00000 1.08883
M_TRISTIMULUS_TO_XYZD65 = np.array([0.95047, 1, 1.08883], dtype=np.float64)


@jit(nopython=True, fastmath=True, cache=True)
def xyzd65_to_lab(c):
    def f_1(t):
        return np.where(t > 0.00885645167903563082, t ** (1.0 / 3.0), 7.78703703703703704 * t + 16.0 / 116.0)

    fx = f_1(c[..., 0] / M_TRISTIMULUS_TO_XYZD65[0])
    fy = f_1(c[..., 1] / M_TRISTIMULUS_TO_XYZD65[1])
    fz = f_1(c[..., 2] / M_TRISTIMULUS_TO_XYZD65[2])
    lab = np.empty(c.shape)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


@jit(nopython=True, fastmath=True, cache=True)
def lab_to_xyzd50(c):
    def f_2(t):
        return np.where(t > 0.20689655172413793103, t ** 3, (t - 16.0 / 116.0) / 7.78703703703703704)

    fy = (c[..., 0] + 16.0) / 116.0
    fx = c[..., 1] / 500.0 + fy
    fz = fy - c[..., 2] / 200.0
    xyzd50 = np.empty(c.shape)
    xyzd50[..., 0] = M_TRISTIMULUS_TO_XYZD50[0] * f_2(fx)
    xyzd50[..., 1] = M_TRISTIMULUS_TO_XYZD50[1] * f_2(fy)
    xyzd50[..., 2] = M_TRISTIMULUS_TO_XYZD50[2] * f_2(fz)
    return xyzd50


@jit(nopython=True, fastmath=True, cache=True)
def lab_to_xyzd65(c):
    def f_2(t):
        return np.where(t > 0.20689655172413793103, t ** 3, (t - 16.0 / 116.0) / 7.78703703703703704)

    fy = (c[..., 0] + 16.0) / 116.0
    fx = c[..., 1] / 500.0 + fy
    fz = fy - c[..., 2] / 200.0
    xyzd65 = np.empty(c.shape)
    xyzd65[..., 0] = M_TRISTIMULUS_TO_XYZD65[0] * f_2(fx)
    xyzd65[..., 1] = M_TRISTIMULUS_TO_XYZD65[1] * f_2(fy)
    xyzd65[..., 2] = M_TRISTIMULUS_TO_XYZD65[2] * f_2(fz)
    return xyzd65


def xyz_to_lab_d50(c):
    xyzd50 = xyzd65_to_xyzd50(c)
    lab = xyzd50_to_lab(c)
    return lab


def xyz_to_lab_d65(c):
    lab = xyzd65_to_lab(c)
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


def labd50_to_xyz(c):
    xyzd50 = lab_to_xyzd50(c)
    xyzd65 = xyzd50_to_xyzd65(xyzd50)
    return xyzd65


def labd65_to_xyz(c):
    return lab_to_xyzd65(c)


def lab_to_xyz(c):
    return lab_to_xyzd65(c)


def srgb_to_xyz(c):
    c = np.array(c)
    lrgb = srgb_to_lrgb(c)
    xyz = lrgb_to_xyz(lrgb)
    return xyz


def srgb_to_lab(c):
    c = np.array(c)
    xyz = srgb_to_xyz(c)
    lab = xyz_to_lab(xyz)
    return lab


def xyz_to_srgb(c):
    c = np.array(c)
    lrgb = xyz_to_lrgb(c)
    srgb = lrgb_to_srgb(lrgb)
    return srgb


def lab_to_srgb(c):
    c = np.array(c)
    xyz = lab_to_xyz(c)
    srgb = xyz_to_srgb(xyz)
    return srgb


def xyz_to_lab(c):
    return xyz_to_lab_d65(c)


# Adapted from https://github.com/lovro-i/CIEDE2000/blob/master/ciede2000.py
# via The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations
# Gaurav Sharma, Wencheng Wu, Edul N. Dalal
@jit(nopython=True, fastmath=True, cache=True)
def _ciede2000(lab_c1, lab_c2):
    L1, a1, b1 = lab_c1[0], lab_c1[1], lab_c1[2]
    L2, a2, b2 = lab_c2[0], lab_c2[1], lab_c2[2]
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_ave ** 7 / (C_ave ** 7 + 6103515625)))
    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2
    C1_ = np.sqrt(a1_ ** 2 + b1_ ** 2)
    C2_ = np.sqrt(a2_ ** 2 + b2_ ** 2)
    if b1_ == 0 and a1_ == 0:
        h1_ = 0
    elif a1_ >= 0:
        h1_ = np.arctan2(b1_, a1_)
    else:
        h1_ = np.arctan2(b1_, a1_) + 2 * np.pi

    if b2_ == 0 and a2_ == 0:
        h2_ = 0
    elif a2_ >= 0:
        h2_ = np.arctan2(b2_, a2_)
    else:
        h2_ = np.arctan2(b2_, a2_) + 2 * np.pi
    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
        dh_ = 0
    elif dh_ > np.pi:
        dh_ -= 2 * np.pi
    elif dh_ < -np.pi:
        dh_ += 2 * np.pi
    dH_ = 2 * np.sqrt(C1_ * C2_) * np.sin(dh_ / 2)
    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2
    _dh = np.abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_
    if _dh <= np.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2
    elif _dh > np.pi and _sh < 2 * np.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 + np.pi
    elif _dh > np.pi and _sh >= 2 * np.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 - np.pi
    else:
        h_ave = h1_ + h2_
    T = 1 - 0.17 * np.cos(h_ave - np.pi / 6) + 0.24 * np.cos(2 * h_ave) + 0.32 * np.cos(
        3 * h_ave + np.pi / 30) - 0.2 * np.cos(4 * h_ave - 63 * np.pi / 180)
    h_ave_deg = h_ave * 180 / np.pi
    if h_ave_deg < 0:
        h_ave_deg += 360
    elif h_ave_deg > 360:
        h_ave_deg -= 360
    dTheta = 30 * np.exp(-(((h_ave_deg - 275) / 25) ** 2))
    R_C = 2 * np.sqrt(C_ave ** 7 / (C_ave ** 7 + 6103515625))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / np.sqrt(20 + Lm50s)
    R_T = -np.sin(dTheta * np.pi / 90) * R_C
    k_L, k_C, k_H = 1, 1, 1
    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    dE_00 = np.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)
    return dE_00


def my_ciede2000(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return _ciede2000(c1, c2)

#
#
# # [ 47.98194319  -3.19681298 -39.3202402 ]
# if __name__ == '__main__':
#     #     # print (np.array([0.12156863, 0.46666667, 0.70588235]))
#     tmp = srgb_to_lrgb(np.array([0.82156863, 0.26666667, 0.30588235]))
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
#     # [ 47.98083149  -3.20353037 -39.33214908]
#     # [49.87586907 56.02126521 25.91319689]
#     tmp = my_ciede2000([47.98083149,  -3.20353037, -39.33214908], [49.87586907, 56.02126521, 25.91319689])
#     print(tmp)
#     tmp = ciede2000((47.98083149,  -3.20353037, -39.33214908), (49.87586907, 56.02126521, 25.91319689))
#
