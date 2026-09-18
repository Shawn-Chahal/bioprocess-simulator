DPI = 600

K_S_GLC = 0.41
K_S_GLN = 2.04
K_S_LAC = 5.00  # Placeholder value
K_I_LAC = 258
K_I_AMM = 7.81

K_X = 25 * 10 ** 4  # cells/mL
X_M = 6 * 10 ** 6  # cells/mL
N = 1.330

K_L = 0.003
K_GI = 0.0103

Y_GLC = 0.50 * 10 ** 6  # cells/mmol glc #49091.1
Y_GLN = 2.00 * 10 ** 6  # cells/mmol gln
Y_GLU = 0.549886 * 10 ** 6  # cells/mmol glu
Y_LAC = 0.8  # mmol_lac/mmol_glc #1.49
Y_AMM = 0.4  # mmol_amm/mmol_gln #1.21

U_MAX = 2.09 / (24 * 3600)  # s-1
U_MAX_LAC = 0.8276 / (24 * 3600) * 0.5091 * 10 ** (-6)
C_LAC_MAX_THEORY = 1000  # mM
C_AMM_MAX_THEORY = 500  # mM
K_D_GLN = 0.02  # mM
K_D_LAC = U_MAX / C_LAC_MAX_THEORY  # 0.001/(24*3600) #s-1*mM-1
K_D_AMM = U_MAX / C_AMM_MAX_THEORY  # 0.006/(24*3600) #s-1*mM-1
K_D_MAX = 0.89 / (24 * 3600)  # s-1 #0.188/3600
K_D = 2.00 * 10 ** (-6)  # s-1
K_DECOMP = 0.086 / (24 * 3600)  # s-1

K_LAG = 6.67026

LAG_TIME = 12 * 3600  # s, duration of lag phase
STEADY_STATE_FRAC = 0.99

DIR_FIGURES = 'figures'
