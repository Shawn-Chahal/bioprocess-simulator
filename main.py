import os

import numpy as np
from matplotlib import pyplot as plt

import source.functions as func
from source.constants import *

ph = 7
temp = 37
c_glc_0 = 25  # mM
c_glc = [c_glc_0]
c_gln_0 = 5  # mM
c_gln = [c_gln_0]
c_lac_0 = 0  # mM
c_lac = [c_lac_0]
c_amm_0 = 0  # mM
c_amm = [c_amm_0]
x_0 = 1.0 * 10 ** 6  # cells/ml
x_v = [x_0]
x_t = [x_0]
t_0 = 0
t = [t_0]
dt = 60  # s
t_f = 3600 * 24 * 21

while t[-1] < t_f:
    k_lag = np.log(1 - STEADY_STATE_FRAC) / (-LAG_TIME)
    alpha = 1 - np.exp(-k_lag * t[-1])

    mu, alpha_sgr = func.specific_growth_rate(ph, temp, c_glc[-1], c_gln[-1], c_lac[-1], c_amm[-1], x_v[-1])
    k_d = func.specific_death_rate(c_lac[-1], c_amm[-1], c_gln[-1])

    d_x_v = alpha * func.ddt_x_v(mu, k_d, x_v[-1])
    d_x_t = alpha * func.ddt_x_t(mu, x_v[-1])
    d_glc = alpha * func.ddt_glc(mu, x_v[-1], alpha_sgr)
    d_gln = alpha * func.ddt_gln(mu, x_v[-1], c_gln[-1])
    d_lac = alpha * func.ddt_lac(c_lac[-1], c_glc[-1], mu, x_v[-1], alpha_sgr)
    d_amm = alpha * func.ddt_amm(mu, x_v[-1], c_gln[-1])

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

fig, axes = plt.subplots(nrows=3, ncols=1, dpi=DPI, figsize=(6.5, 12), layout="constrained")
line_style = dict(linestyle='-', marker='none')  # marker = 'o',  ms = 0.5, mec = '#000000'

t_plot = np.array(t) / (24 * 3600)
x_v_plot = np.array(x_v) / (10 ** 6)
x_t_plot = np.array(x_t) / (10 ** 6)

axes[0].plot(t_plot, c_glc, label=r"$C_{glc}$", **line_style)
axes[0].plot(t_plot, c_gln, label=r"$C_{gln}$", **line_style)
axes[0].plot(t_plot, c_lac, label=r"$C_{lac}$", **line_style)
axes[0].plot(t_plot, c_amm, label=r"$C_{amm}$", **line_style)
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

fig.savefig(os.path.join(DIR_FIGURES, 'Euler.png'))
plt.close(fig)
