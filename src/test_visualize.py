import numpy as np
from classes_CHG4901 import Bioprocess

model = Bioprocess()
print("Paramètres :", model.k_psi, model.k_p, model.f_inf, model.k_inact, model.k_it_toxic)

# Test parameters (not scientifically calibrated)
model.u_inf = 1.0


# Run simulation for 24 hours
model.simulate(t_f=10 * 24 * 3600, dt=60)

# Generate the four graphs
model.visualize("CHG4901_results.png", dpi=150)

print("Simulation completed!")
print("Figure saved: CHG4901_results.png")

t_day = np.array(model.t) / 86400
i = np.argmin(np.abs(t_day - 6))

viability = model.x_v[i] / model.x_t[i]

print("Day 6 results:")
print("Viability:", viability * 100, "%")
print("Total LV:", model.p_lv_tot[i])
print("Infectious LV:", model.p_lv_inf[i])
