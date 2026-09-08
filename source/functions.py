import numpy as np
from scipy import optimize

from source.constants import *


def f_ph(ph, sigma, mu):
    return np.exp(-(ph - mu) ** 2 / (2 * sigma ** 2))


def f_temp(temp, b, c, temp_min, temp_max):
    res = optimize.minimize_scalar(neg_u_temp, args=(b, c, temp_min, temp_max), bounds=(temp_min, temp_max))
    norm_u_temp = u_temp(temp, b, c, temp_min, temp_max) / (-res.fun)
    norm_u_temp = np.where(temp > temp_max, 0, norm_u_temp)
    return norm_u_temp


def u_temp(temp, b, c, temp_min, temp_max):
    u = ((b * (temp - temp_min)) ** 2) * (1 - np.exp(c * (temp - temp_max)))
    return u


def neg_u_temp(temp, b, c, temp_min, temp_max):
    return -u_temp(temp, b, c, temp_min, temp_max)


def f_glc(c_glc):
    return c_glc / (K_SS + c_glc)


def f_gln(c_gln):
    return c_gln / (K_SN + c_gln)


def f_lac(c_lac, c_glc, mode=1):
    # print(K_IL / (K_IL + c_lac), lac_switch(c_lac, c_glc))
    if mode == 1:
        return K_IL / (K_IL + c_lac)
    elif mode == 2:
        return (K_IL / (K_IL + c_lac)) * lac_switch(c_lac, c_glc)
    else:
        return None


def f_amm(c_amm):
    return K_IM / (K_IM + c_amm)


def f_cd(x_v):
    if x_v > X_M:
        return 0
    else:
        return (x_v / (K_X + x_v)) * (1 - x_v / X_M) ** N


def beta(c_glc):
    if c_glc > 5.4:
        b = 0
    else:
        b = 1
    return b


def lac_switch(c_lac, c_glc):
    return (c_lac / (K_L + c_lac)) * (K_GI / (K_GI + c_glc))


def spec_growth(ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=2):
    u_ph = f_ph(ph, 1, 7)
    norm_u_temp = f_temp(temp, 0.035, 0.7, 0, 40)
    u_glc = f_glc(c_glc)
    u_gln = f_gln(c_gln)
    u_lac = f_lac(c_lac, c_glc)
    u_amm = f_amm(c_amm)
    u_cd = f_cd(x_v)
    if mode == 1:
        return U_MAX * u_ph * norm_u_temp * u_glc * u_gln * u_lac * u_amm * u_cd
    elif mode == 2:
        return U_MAX * u_ph * norm_u_temp * (u_glc + u_gln) * u_lac * u_amm * u_cd
    elif mode == 3:
        if c_glc > 0.4:
            return U_MAX * u_ph * norm_u_temp * (u_glc + u_gln) * u_lac * u_amm * u_cd
        else:
            return U_MAX * u_ph * norm_u_temp * (u_glc + u_gln + u_lac) * u_amm * u_cd
    else:
        return None


def spec_death(c_lac, c_amm, c_gln, mode=1):
    if mode == 1:
        return K_D
    elif mode == 2:
        return K_D_MAX * (K_D_GLN / (K_D_GLN + c_gln))
    elif mode == 3:
        return K_D_MAX * (K_D_GLN / (K_D_GLN + c_gln)) / ((U_MAX - K_D_LAC * c_lac) * (U_MAX - K_D_AMM * c_amm))
    else:
        return None


def ddt_x_g(mu, x_v):
    return mu * x_v


def ddt_x_d(k_d, x_v):
    return k_d * x_v


def ddt_x_v(mu, k_d, x_v):
    return ddt_x_g(mu, x_v) - ddt_x_d(k_d, x_v)


def ddt_x_t(mu, x_v):
    return ddt_x_g(mu, x_v)


def ddt_glc(mu, x_v):
    d_glc = -(mu / Y_GLC) * x_v
    return d_glc


def ddt_gln(mu, x_v, c_gln):
    d_gln = -(mu / Y_GLN) * x_v - K_DECOMP * c_gln
    return d_gln


def ddt_lac(c_lac, c_glc, mu, x_v, mode=3):
    if mode == 1:
        return Y_LAC * (-ddt_glc(mu, x_v))
    elif mode == 2:
        return (1 - beta(c_glc)) * Y_LAC * (-ddt_glc(mu, x_v)) - beta(c_glc) * 0.05 * Y_LAC * (U_MAX / Y_GLC) * x_v
    elif mode == 3:
        return Y_LAC * (-ddt_glc(mu, x_v)) - K_52 * U_MAX_2 * lac_switch(c_lac, c_glc)
    else:
        return None


def ddt_amm(mu, x_v, c_gln):
    d_amm = Y_AMM * (-ddt_gln(mu, x_v, c_gln)) + K_DECOMP * c_gln
    return d_amm
