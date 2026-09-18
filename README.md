# Bioprocess simulator

## Summary

This is a mechanistic model for mammalian cell growth. It monitors cell density and metabolite concentration over time.

The `main.py` folder is where this simulation operates. Initial conditions are stated at the beginning. This includes
the pH, temperature, metabolite concentrations, cell density and the total duration of this simulation. After each
initial condition is stated, a corresponding array is created to store the changing data, to later graph each
parameter's overall change. Below is an example using the initial glucose concentration.

```
c_glc_0 = 25  # mM
c_glc = [c_glc_0]
```

A while loop is then created, to run Euler's method with these conditions as a bioprocess simulator, until the stated
final time is reached. Within this while loop, the most recent term in each array will be used (ex. `t[-1]`) since
Euler's method is recursive. Initially, `k_lag` is defined and used in an alpha term which will then be used to model a
lag phase in the growth of the cell culture. Adjustment of the `LAG_TIME` constant will adjust the lag time seen in the
graph.

```
    k_lag = np.log(1 - STEADY_STATE_FRAC) / (-LAG_TIME)
    alpha = 1 - np.exp(-k_lag * t[-1])
```

The specific growth rate and specific death rate functions are then passed as terms `mu` and `k_d` respectively, to be
used in the differential equations. A series of equations are then used to determine the rate of change of each of the
varying parameters (the metabolite concentrations and cell density) by calling the differential equations from
`functions.py`. The alpha term is multiplied in here to model the initial lag phase. Below is an example using the
glutamine term.

```
d_gln = alpha * func.ddt_gln(mu, x_v[-1], c_gln[-1])
```

The series of `'parameter'_next` functions take the latest term of each parameter and add it to the product of the
previously calculated slope and the change in time. This value is stored as the 'next' value of each parameter. The
maximum function prevents metabolite or cell density values from dropping below zero, as it is unrealistic. These new
values are then added to their respective arrays which contain the change of each parameter at every time step. Below is
an example using the lactate term.

```
c_lac_next = max(0, c_lac[-1] + d_lac * dt)
c_lac.append(c_lac_next)
```

The rest of `main.py` is formatting for generating the graphs. Subplots are created to fit three graphs on one figure.
The dimensions are stated, and a dictionary is created for the line style of the graphs. The time array is converted
from seconds to days, and the various cell densities are divided in terms of $10^6$ cells/mL for increased clarity in
terms of axis units and scaling. The first plot is of the metabolite concentrations over time. The second is for the
viable cell density (VCD) and total cell density (TCD). Lastly, the last graph plots the cell viability by dividing the
viable cell density by the total cell density. The data of the graphs can be manipulated by not only altering the
functions in `functions.py` but also by altering the constants in `constants.py`. Below is the resulting figure.

![Euler method](figures/Euler.png)

---

## References (explanation of utility)

### Functions

This section is a breakdown of `functions.py`, what each function does and its reference (link and equation number).

**Note: All `f_'blank'` functions are normalized between 0 and 1**

`f_ph(ph, sigma, mu)`: Effect of pH on specific growth rate. Normalized normal distribution function with the center set
at 7.0 (optimized cell growth at a pH of 7.0).

`f_temp(temp, b, c, temp_min, temp_max)`: (Normalized) effect of temperature on specific growth rate. For this function
to be normalized, two other functions are used, `u_temp` and `neg_u_temp`.

`u_temp(temp, b, c, temp_min, temp_max)`: Effect of temperature on specific growth rate.
([Noll et al. 2020](https://doi.org/10.3390/pr8010121), Eq. 23)

`neg_u_temp(temp, b, c, temp_min, temp_max)`: Negative of `u_temp` function.

`f_glc(c_glc)`: Effect of glucose on specific growth rate.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 37),
([Martens et al. 1995](https://doi.org/10.1002/bit.260480109), Eq. 50)

`f_gln(c_gln)`: Effect of glutamine on specific growth rate.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 37),
([Martens et al. 1995](https://doi.org/10.1002/bit.260480109), Eq. 50)

`f_lac(c_lac, c_glc, mode=1)`: Effect of lactate on specific growth rate.
([Martens et al. 1995](https://doi.org/10.1002/bit.260480109), Eq. 50)

`f_lac(c_lac, c_glc, mode=2)`: Effect of lactate on specific growth rate incorporating `lac_switch`.
([Martens et al. 1995](https://doi.org/10.1002/bit.260480109), Eq. 50),
([Pimentel et al. 2023](https://doi.org/10.1109/CoDIT58514.2023.10284210), Eq. 3b)

`f_amm(c_amm)`: Effect of ammonia on specific growth rate.
([Martens et al. 1995](https://doi.org/10.1002/bit.260480109), Eq. 50)

`f_cd(x_v)`: Effect of cell density on specific growth rate.
([Lee 2002](https://doi-org.proxy.bib.uottawa.ca/10.1007/BF02935890), Eq. 4)

`beta(c_glc)`: Enables a switch between inactive (0) to active (1) when glucose levels reach a threshold.

`lac_switch(c_lac, c_glc)`: Enables a metabolic switch at low enough glucose levels, in which lactate acts as a nutrient
to the cells instead of an inhibitor. ([Pimentel et al. 2023](https://doi.org/10.1109/CoDIT58514.2023.10284210), Eq. 3b)

`spec_growth(ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=1)`: Specific growth rate of cells.

`spec_growth(ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=2)`: Specific growth rate of cells (glucose consumption
continuing after glutamine depletion).

`spec_growth(ph, temp, c_glc, c_gln, c_lac, c_amm, x_v, mode=3)`: Specific growth rate of cells (lactate consumption
after glucose depletion).

`spec_death(c_lac, c_amm, c_gln, mode=1)`: Specific death rate of cells as a constant.

`spec_death(c_lac, c_amm, c_gln, mode=2)`: Specific death rate of cells incorporating glutamine concentration.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 25)

`spec_death(c_lac, c_amm, c_gln, mode=3)`: Specific death rate of cells incorporating glutamine, lactate and ammonia
concentrations. ([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 38)

`ddt_x_g(mu, x_v)`: The definition of cell growth.

`ddt_x_d(k_d, x_v)`: The definition of cell death.

`ddt_x_v(mu, k_d, x_v)`: The definition of viable cell density.

`ddt_x_t(mu, x_v)`: The definition of total cell density (accumulation of cell growth).

`ddt_glc(mu, x_v)`: Time derivative of glucose concentration.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 26, Eq. 30)

`ddt_gln(mu, x_v, c_gln)`: Time derivative of glutamine concentration.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 31, Eq. 40),
([Kurano et al. 1989](https://doi.org/10.1016/0168-1656(90)90055-G), Eq. 7)

`ddt_lac(c_lac, c_glc, mu, x_v, mode=1)`: Time derivative of lactate concentration.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 32, Eq. 41)

`ddt_lac(c_lac, c_glc, mu, x_v, mode=2)`: Time derivative of lactate concentration with metabolic switch.
([Gadgil 2014](https://doi.org/10.1002/jctb.4302), Eq. 5b)

`ddt_lac(c_lac, c_glc, mu, x_v, mode=3)`: Time derivative of lactate concentration with metabolic switch.
([Pimentel et al. 2023](https://doi.org/10.1109/CoDIT58514.2023.10284210), Eq. 2e, Eq. 3b)

`ddt_amm(mu, x_v, c_gln)`: Time derivative of ammonia concentration.
([de Tremblay et al. 1993](https://doi.org/10.1007/BF00389535), Eq. 33, Eq. 42),
([Kurano et al. 1989](https://doi.org/10.1016/0168-1656(90)90055-G), Eq. 7)

### Constants

*This is a complete list of constants with values taken from literature.*

```
K_SS = 0.41
K_SN = 2.04
K_IL = 258
K_IM = 7.81
```

([Martens et al. 1995](https://doi.org/10.1002/bit.260480109))

```
K_X = 25 * 10 ** 4
N = 1.330
```

([Lee 2002](https://doi-org.proxy.bib.uottawa.ca/10.1007/BF02935890))

```
K_L = 0.003
K_GI = 0.0103
K_52 = 0.5091
U_MAX_2 = 0.8276 / (24 * 3600)
```

([Pimentel et al. 2023](https://doi.org/10.1109/CoDIT58514.2023.10284210))

```
K_D_GLN = 0.02
```

([de Tremblay et al. 1992](https://doi.org/10.1007/BF00369551))

```
K_DECOMP = 0.086 / (24 * 3600)
```

([Kurano et al. 1989](https://doi.org/10.1016/0168-1656(90)90055-G))
