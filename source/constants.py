K_S_GLC = 0.41
K_S_GLN = 2.04
K_S_LAC = 1.00
K_I_LAC = 258
K_I_AMM = 7.81

K_X = 25 * 10 ** 4  # cells/mL
X_M = 6 * 10 ** 6  # cells/mL
N = 1.330

K_ALPHA = 1.0

U_MAX = 2.09 / (24 * 3600)  # s-1
U_MAX_LAC = 0.8276 / (24 * 3600) * 0.5091 * 10 ** (-6)

K_D = 2.00 * 10 ** (-6)  # s-1
K_DECOMP = 0.086 / (24 * 3600)  # s-1

K_D_MAX = 0.89 / (24 * 3600)  # s-1
K_D_GLN = 0.02  # mM
K_D_LAC = 0.00209 / (24 * 3600)  # s-1*mM-1
K_D_AMM = 0.00418 / (24 * 3600)  # s-1*mM-1

K_L = 0.003
K_GI = 0.0103

Y_X_GLC = 0.40 * 10 ** 6  # cells/mmol glc
Y_X_GLN = 6.00 * 10 ** 6  # cells/mmol gln
Y_X_LAC = 0.05 * 10 ** 6  # cells/mmol lac
Y_LAC_GLC = 1.49  # mmol_lac/mmol_glc
Y_AMM_GLN = 0.8  # mmol_amm/mmol_gln

LAG_TIME = 12 * 3600  # s, duration of lag phase
STEADY_STATE_FRAC = 0.99

DPI = 600
DIR_FIGURES = 'figures'
