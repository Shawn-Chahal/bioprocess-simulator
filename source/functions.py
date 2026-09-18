import numpy as np
from scipy.special import lambertw

from source.constants import *


def f_ph(ph, ph_opt, ph_sd):
    return np.exp(-(ph - ph_opt) ** 2 / (2 * ph_sd ** 2))


def f_temp(temp, k_t, temp_min, temp_max):
    # TODO: Replace k_t with temp_opt and reparameterize
    z = 2 * np.exp(2 + k_t * (temp_max - temp_min))
    w = np.real(lambertw(z))
    t_opt = temp_min + (w - 2) / k_t
    f_temp_max = ((t_opt - temp_min) ** 2) * (1 - np.exp(k_t * (t_opt - temp_max)))

    f_temp_ = ((temp - temp_min) ** 2) * (1 - np.exp(k_t * (temp - temp_max)))
    f_temp_ = f_temp_ / f_temp_max
    f_temp_ = np.where(temp > temp_min, f_temp_, 0)
    f_temp_ = np.where(temp < temp_max, f_temp_, 0)

    return f_temp_


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
    f_ph_ = f_ph(ph, ph_opt=7, ph_sd=1)
    f_temp_ = f_temp(temp, k_t=0.7, temp_min=0, temp_max=40)
    u_glc = f_glc(c_glc)
    u_gln = f_gln(c_gln)
    u_lac = f_lac(c_lac, c_glc)
    u_amm = f_amm(c_amm)
    u_cd = f_cd(x_v)
    if mode == 1:
        return U_MAX * f_ph_ * f_temp_ * u_glc * u_gln * u_lac * u_amm * u_cd
    elif mode == 2:
        return U_MAX * f_ph_ * f_temp_ * (u_glc + u_gln) * u_lac * u_amm * u_cd
    elif mode == 3:
        if c_glc > 0.4:
            return U_MAX * f_ph_ * f_temp_ * (u_glc + u_gln) * u_lac * u_amm * u_cd
        else:
            return U_MAX * f_ph_ * f_temp_ * (u_glc + u_gln + u_lac) * u_amm * u_cd
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
