# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:15:35 2021

"""

import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import Rbf, interp1d

# Train the model for the gas turbine map
gt_map = pd.read_csv("data/gt_map.csv")
gt_map.drop_duplicates(inplace=True)

# Normalize data
ds, ts = MinMaxScaler(), MinMaxScaler()
x = ds.fit_transform(gt_map[["alt", "ma", "power"]].to_numpy())
y = ts.fit_transform(gt_map[["fuelFlow", "eff"]].to_numpy())

# RBF
r_flf = Rbf(x[:, 0], x[:, 1], x[:, 2], y[:, 0])
r_eff = Rbf(x[:, 0], x[:, 1], x[:, 2], y[:, 1])


NOx = [
    [0.1337e2, 0.9144e-1, 0.3617e-4, -0.1075e1, -0.6473e0, 0.2994e0],
    [0.7194e1, 0.5609e0, -0.1059e-1, -0.3223e1, 0.2889e0, 0.2591e0],
    [0.3699e0, 0.5470e0, -0.7445e-2, -0.6914e1, 0.6782e1, 0.1138e0],
    [0.1605e0, 0.2412e0, -0.1650e-2, -0.8818e1, 0.3714e2, -0.2268e0],
]

CO = [
    [-0.4312e0, 0.9648e-1, -0.2328e-2, -0.5474e0, -0.1131e0, 0.2726e-1],
    [0.1788e0, 0.5469e-1, -0.1710e-2, -0.5147e0, -0.2318e0, 0.3439e-1],
    [0.8187e1, -0.2967e0, 0.5588e-2, 0.3856e1, 0.4757e1, -0.3191e0],
    [0.2051e2, 0.1776e1, -0.4778e-1, -0.1390e3, 0.1528e3, -0.2915e1],
]

# Data extracted from P127 TM model using FP50 mission
FF = [0.1544435, 0.1399368, 0.0564546, 0.0397422]

PR = 14.8

EINO, EICO = [], []


for i in range(len(FF)):
    ff = FF[i]
    k_nox, k_co = NOx[i], CO[i]

    EINO.append(
        k_nox[0]
        + k_nox[1] * PR
        + k_nox[2] * PR**2
        + k_nox[3] * ff
        + k_nox[4] * ff**2
        + k_nox[5] * ff * PR
    )
    EICO.append(
        k_co[0]
        + k_co[1] * PR
        + k_co[2] * PR**2
        + k_co[3] * ff
        + k_co[4] * ff**2
        + k_co[5] * ff * PR
    )

FF.reverse()
EINO.reverse()
EICO.reverse()

f_eino = interp1d(FF, EINO, fill_value="extrapolate")
f_eico = interp1d(FF, EICO, fill_value="extrapolate")


def engine_map(alt, mach, power):
    # print(alt, mach, power)
    x = list(ds.transform([[alt, mach, power]]).ravel())
    y = [r_flf(*x), r_eff(*x)]
    out = ts.inverse_transform(np.atleast_2d(y))

    EI_no, EI_co = EI(alt, out[0, 0], mach)

    f_NO, f_CO = EI_no * out[0, 0], EI_co * out[0, 0]

    return out[0, 0], out[0, 1], f_NO, f_CO


def EI(alt, ff, mach):
    # ISA Atmosphere Mach

    # Temperature in Kelvin
    if alt < 11000:
        T = 288.15 - alt * 0.0065
        P = 101.3e3 * (T / 288.15) ** (9.8 / (0.0065 * 287))
    else:
        T = 216.65
        P = 22643.217 * np.e ** (9.8 * (11000 - alt) / (T * 287))

    theta = T / 288.15
    delta = P / 101325

    ff_sl = ff * (theta**4.5 / delta) * np.e ** (0.2 * (mach**2))

    EIno_sl = f_eino(ff_sl)
    EIco_sl = f_eico(ff_sl)

    EIno = EIno_sl * (delta**1.02 / theta**3.3) ** 0.5
    EIco = EIco_sl * (theta**3.3 / delta**1.02) ** 1.0

    return EIno, EIco
