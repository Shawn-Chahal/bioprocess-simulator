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


def f_s_glc(c_glc):
    return c_glc / (K_S_GLC + c_glc)


def f_s_gln(c_gln):
    return c_gln / (K_S_GLN + c_gln)


def f_s_lac(c_lac):
    return c_lac / (K_S_LAC + c_lac)


def f_i_lac(c_lac):
    return K_I_LAC / (K_I_LAC + c_lac)


def f_i_amm(c_amm):
    return K_I_AMM / (K_I_AMM + c_amm)


def f_x(x_v):
    x_v_ = np.clip(x_v, 0, X_M)
    return (x_v_ / (K_X + x_v_)) * (1 - x_v_ / X_M) ** N


def specific_growth_rate(ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=2):
    f_ph_ = f_ph(ph, ph_opt=7, ph_sd=1)
    f_temp_ = f_temp(temp, k_t=0.7, temp_min=0, temp_max=40)
    f_s_glc_ = f_s_glc(c_glc)
    f_s_gln_ = f_s_gln(c_gln)
    f_s_lac_ = f_s_lac(c_lac)
    f_i_lac_ = f_i_lac(c_lac)
    f_i_amm_ = f_i_amm(c_amm)
    f_x_ = f_x(x_v)

    f_env = f_ph_ * f_temp_ * f_i_lac_ * f_i_amm_ * f_x_
    alpha_sgr = None

    if mode == 1:
        f_s = f_s_glc_ * f_s_gln_
    elif mode == 2:
        alpha_sgr = 1 - np.exp(-K_ALPHA * c_glc)
        f_s_1 = f_s_glc_ * f_s_gln_
        f_s_2 = f_s_lac_ * f_s_gln_
        f_s = alpha_sgr * f_s_1 + (1 - alpha_sgr) * f_s_2
    else:
        f_s = 0

    return U_MAX * f_s * f_env, alpha_sgr


def specific_death_rate(c_lac, c_amm, c_gln, mode=1):
    if mode == 1:
        return K_D
    elif mode == 2:
        return K_D_MAX * (K_D_GLN / (K_D_GLN + c_gln))
    elif mode == 3:
        return K_D_MAX * (K_D_GLN / (K_D_GLN + c_gln)) / ((U_MAX - K_D_LAC * c_lac) * (U_MAX - K_D_AMM * c_amm))
    else:
        return None


def ddt_x_v(mu, k_d, x_v):
    return mu * x_v - k_d * x_v


def ddt_x_t(mu, x_v):
    return mu * x_v


def ddt_glc(mu, x_v, alpha_sgr=None):
    if alpha_sgr is not None:
        q_glc = alpha_sgr * mu / Y_X_GLC
    else:
        q_glc = mu / Y_X_GLC

    return - q_glc * x_v


def ddt_gln(mu, x_v, c_gln):
    q_gln = mu / Y_X_GLN
    return -q_gln * x_v - K_DECOMP * c_gln


def ddt_lac(c_lac, c_glc, mu, x_v, alpha_sgr=None, mode=2):
    if alpha_sgr is not None:
        q_glc = alpha_sgr * mu / Y_X_GLC
        q_lac = (1 - alpha_sgr) * mu / Y_X_LAC
    else:
        q_glc = mu / Y_X_GLC
        q_lac = 0

    if mode == 1:
        return Y_LAC_GLC * q_glc * x_v
    elif mode == 2:
        return Y_LAC_GLC * q_glc * x_v - q_lac * x_v
    elif mode == 3:
        return Y_LAC_GLC * q_glc * x_v - U_MAX_LAC * (c_lac / (K_L + c_lac)) * (K_GI / (K_GI + c_glc)) * x_v
    else:
        return None


def ddt_amm(mu, x_v, c_gln):
    q_gln = mu / Y_X_GLN
    return Y_AMM_GLN * q_gln * x_v + K_DECOMP * c_gln
