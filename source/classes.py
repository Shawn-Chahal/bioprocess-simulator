from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import lambertw


@dataclass
class Bioprocess:
    k_s_glc: float = 0.41  # mM
    k_s_gln: float = 2.04  # mM
    k_s_lac: float = 1.00  # mM
    k_i_lac: float = 258.0  # mM
    k_i_amm: float = 7.81  # mM

    k_x: float = 0.25 * 10 ** 6  # cells/mL
    x_m: float = 6 * 10 ** 6  # cells/mL
    n: float = 1.330

    k_alpha: float = 1.0

    u_max: float = 2.09 / (24 * 3600)  # s^-1
    u_max_lac: float = 0.8276 / (24 * 3600) * 0.5091 * 10 ** (-6)

    k_d_0: float = 2.00 * 10 ** (-6)  # s^-1
    k_decomp: float = 0.086 / (24 * 3600)  # s^-1

    k_d_max: float = 0.89 / (24 * 3600)  # s^-1
    k_d_gln: float = 0.02
    k_d_lac: float = 0.00209 / (24 * 3600)  # s^-1*mM^-1
    k_d_amm: float = 0.00418 / (24 * 3600)  # s^-1*mM^-1

    k_l: float = 0.003
    k_gi: float = 0.0103

    y_x_glc: float = 0.40 * 10 ** 6  # cells/mmol glc
    y_x_gln: float = 6.00 * 10 ** 6  # cells/mmol gln
    y_x_lac: float = 0.05 * 10 ** 6  # cells/mmol lac

    y_lac_glc: float = 1.49  # mmol_lac/mmol_glc
    y_amm_gln: float = 0.8  # mmol_amm/mmol_gln

    lag_time: float = 12 * 3600  # s, duration of lag phase
    steady_state_frac: float = 0.99

    ph_opt: float = 7.0
    ph_sd: float = 1.0

    k_t: float = 0.7
    temp_min: float = 0
    temp_max: float = 40

    x_v: Optional[np.ndarray] = field(init=False, default=None)
    x_t: Optional[np.ndarray] = field(init=False, default=None)
    c_glc: Optional[np.ndarray] = field(init=False, default=None)
    c_gln: Optional[np.ndarray] = field(init=False, default=None)
    c_lac: Optional[np.ndarray] = field(init=False, default=None)
    c_amm: Optional[np.ndarray] = field(init=False, default=None)
    t: Optional[np.ndarray] = field(init=False, default=None)

    def simulate(
            self,
            ph=7,
            temp=37,  # C
            c_glc_0=25,  # mM
            c_gln_0=5,  # mM
            c_lac_0=0,  # mM
            c_amm_0=0,  # mM
            x_0=1.0 * 10 ** 6,  # cells/ml
            dt=60,  # s
            t_f=3600 * 24 * 21  # s
    ):

        t_0 = 0

        c_glc = [c_glc_0]
        c_gln = [c_gln_0]
        c_lac = [c_lac_0]
        c_amm = [c_amm_0]
        x_v = [x_0]
        x_t = [x_0]
        t = [t_0]

        while t[-1] < t_f:
            # TODO: Model lag phase better
            k_lag = np.log(1 - self.steady_state_frac) / (-self.lag_time)
            alpha = 1 - np.exp(-k_lag * t[-1])

            mu, alpha_sgr = self.specific_growth_rate(ph, temp, c_glc[-1], c_gln[-1], c_lac[-1], c_amm[-1], x_v[-1])
            k_d = self.specific_death_rate(c_lac[-1], c_amm[-1], c_gln[-1])

            d_x_v = alpha * self.ddt_x_v(mu, k_d, x_v[-1])
            d_x_t = alpha * self.ddt_x_t(mu, x_v[-1])
            d_glc = alpha * self.ddt_glc(mu, x_v[-1], alpha_sgr)
            d_gln = alpha * self.ddt_gln(mu, x_v[-1], c_gln[-1])
            d_lac = alpha * self.ddt_lac(c_lac[-1], c_glc[-1], mu, x_v[-1], alpha_sgr)
            d_amm = alpha * self.ddt_amm(mu, x_v[-1], c_gln[-1])

            # Euler's method
            x_v_next = max(0, x_v[-1] + d_x_v * dt)
            x_t_next = max(0, x_t[-1] + d_x_t * dt)
            c_glc_next = max(0, c_glc[-1] + d_glc * dt)
            c_gln_next = max(0, c_gln[-1] + d_gln * dt)
            c_lac_next = max(0, c_lac[-1] + d_lac * dt)
            c_amm_next = max(0, c_amm[-1] + d_amm * dt)

            x_v.append(x_v_next)
            x_t.append(x_t_next)
            c_glc.append(c_glc_next)
            c_gln.append(c_gln_next)
            c_lac.append(c_lac_next)
            c_amm.append(c_amm_next)
            t.append(t[-1] + dt)

        self.x_v = np.array(x_v)
        self.x_t = np.array(x_t)
        self.c_glc = np.array(c_glc)
        self.c_gln = np.array(c_gln)
        self.c_lac = np.array(c_lac)
        self.c_amm = np.array(c_amm)
        self.t = np.array(t)

    def visualize(self, filepath, dpi=600):
        fig, axes = plt.subplots(nrows=3, ncols=1, dpi=dpi, figsize=(6.5, 12), layout="constrained")
        line_style = dict(linestyle='-', marker='none')  # marker = 'o',  ms = 0.5, mec = '#000000'

        t_plot = np.array(self.t) / (24 * 3600)
        x_v_plot = np.array(self.x_v) / (10 ** 6)
        x_t_plot = np.array(self.x_t) / (10 ** 6)

        axes[0].plot(t_plot, self.c_glc, label=r"$C_{glc}$", **line_style)
        axes[0].plot(t_plot, self.c_gln, label=r"$C_{gln}$", **line_style)
        axes[0].plot(t_plot, self.c_lac, label=r"$C_{lac}$", **line_style)
        axes[0].plot(t_plot, self.c_amm, label=r"$C_{amm}$", **line_style)
        axes[0].legend(fontsize=8)
        axes[0].set_xlabel('Time [day]')
        axes[0].set_ylabel('Concentration [mM]')

        axes[1].plot(t_plot, x_v_plot, label="VCD", **line_style)
        axes[1].plot(t_plot, x_t_plot, label="TCD", **line_style)
        axes[1].set_xlabel('Time [day]')
        axes[1].set_ylabel(r'Cell Density [$10^6$ cells/mL]')
        axes[1].legend(fontsize=8)

        axes[2].plot(t_plot, x_v_plot / x_t_plot, **line_style)
        axes[2].set_xlabel('Time [day]')
        axes[2].set_ylabel('Cell Viability')

        fig.savefig(filepath)
        plt.close(fig)

    def f_ph(self, ph):
        return np.exp(-(ph - self.ph_opt) ** 2 / (2 * self.ph_sd ** 2))

    def f_temp(self, temp):
        # TODO: Replace k_t with temp_opt and reparameterize
        z = 2 * np.exp(2 + self.k_t * (self.temp_max - self.temp_min))
        w = np.real(lambertw(z))
        t_opt = self.temp_min + (w - 2) / self.k_t
        f_temp_max = ((t_opt - self.temp_min) ** 2) * (1 - np.exp(self.k_t * (t_opt - self.temp_max)))

        f_temp_ = ((temp - self.temp_min) ** 2) * (1 - np.exp(self.k_t * (temp - self.temp_max)))
        f_temp_ = f_temp_ / f_temp_max
        f_temp_ = np.where(temp > self.temp_min, f_temp_, 0)
        f_temp_ = np.where(temp < self.temp_max, f_temp_, 0)

        return f_temp_

    def f_s_glc(self, c_glc):
        return c_glc / (self.k_s_glc + c_glc)

    def f_s_gln(self, c_gln):
        return c_gln / (self.k_s_gln + c_gln)

    def f_s_lac(self, c_lac):
        return c_lac / (self.k_s_lac + c_lac)

    def f_i_lac(self, c_lac):
        return self.k_i_lac / (self.k_i_lac + c_lac)

    def f_i_amm(self, c_amm):
        return self.k_i_amm / (self.k_i_amm + c_amm)

    def f_x(self, x_v):
        x_v_ = np.clip(x_v, 0, self.x_m)
        return (x_v_ / (self.k_x + x_v_)) * (1 - x_v_ / self.x_m) ** self.n

    def specific_growth_rate(self, ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=2):
        f_ph_ = self.f_ph(ph)
        f_temp_ = self.f_temp(temp)
        f_s_glc_ = self.f_s_glc(c_glc)
        f_s_gln_ = self.f_s_gln(c_gln)
        f_s_lac_ = self.f_s_lac(c_lac)
        f_i_lac_ = self.f_i_lac(c_lac)
        f_i_amm_ = self.f_i_amm(c_amm)
        f_x_ = self.f_x(x_v)

        f_env = f_ph_ * f_temp_ * f_i_lac_ * f_i_amm_ * f_x_
        alpha_sgr = None

        if mode == 1:
            f_s = f_s_glc_ * f_s_gln_
        elif mode == 2:
            alpha_sgr = 1 - np.exp(-self.k_alpha * c_glc)
            f_s_1 = f_s_glc_ * f_s_gln_
            f_s_2 = f_s_lac_ * f_s_gln_
            f_s = alpha_sgr * f_s_1 + (1 - alpha_sgr) * f_s_2
        else:
            f_s = 0

        return self.u_max * f_s * f_env, alpha_sgr

    def specific_death_rate(self, c_lac, c_amm, c_gln, mode=1):
        if mode == 1:
            return self.k_d_0
        elif mode == 2:
            return self.k_d_max * (self.k_d_gln / (self.k_d_gln + c_gln))
        elif mode == 3:
            a = self.k_d_max * (self.k_d_gln / (self.k_d_gln + c_gln))
            return a / ((self.u_max - self.k_d_lac * c_lac) * (self.u_max - self.k_d_amm * c_amm))
        else:
            return None

    def ddt_x_v(self, mu, k_d, x_v):
        return (mu - k_d) * x_v

    def ddt_x_t(self, mu, x_v):
        return mu * x_v

    def ddt_glc(self, mu, x_v, alpha_sgr=None):
        if alpha_sgr is not None:
            q_glc = alpha_sgr * mu / self.y_x_glc
        else:
            q_glc = mu / self.y_x_glc

        return - q_glc * x_v

    def ddt_gln(self, mu, x_v, c_gln):
        q_gln = mu / self.y_x_gln
        return -q_gln * x_v - self.k_decomp * c_gln

    def ddt_lac(self, c_lac, c_glc, mu, x_v, alpha_sgr=None, mode=2):
        if alpha_sgr is not None:
            q_glc = alpha_sgr * mu / self.y_x_glc
            q_lac = (1 - alpha_sgr) * mu / self.y_x_lac
        else:
            q_glc = mu / self.y_x_glc
            q_lac = 0

        if mode == 1:
            return self.y_lac_glc * q_glc * x_v
        elif mode == 2:
            return self.y_lac_glc * q_glc * x_v - q_lac * x_v
        elif mode == 3:
            a = self.y_lac_glc * q_glc * x_v
            return a - self.u_max_lac * (c_lac / (self.k_l + c_lac)) * (self.k_gi / (self.k_gi + c_glc)) * x_v
        else:
            return None

    def ddt_amm(self, mu, x_v, c_gln):
        q_gln = mu / self.y_x_gln
        return self.y_amm_gln * q_gln * x_v + self.k_decomp * c_gln
