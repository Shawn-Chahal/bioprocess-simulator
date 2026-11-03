
from classes_CHG4901 import Bioprocess

model = Bioprocess()

# Test values (not calibrated parameters)
model.u_inf = 1.0
model.k_psi = 1e-5
model.k_p = 0.001

result = model.ddt_p_lv_tot(
    x_v=1e6,
    p_lv_tot=100
)

print("Calculated g2 =", result)
print("Expected g2 = 9.9")

assert abs(result - 9.9) < 1e-9
print("Test passed!")


# Test of Euler's method
dt = 60
p_lv_tot_initial = 100

p_lv_tot_next = max(0, p_lv_tot_initial + result * dt)

print("Euler result =", p_lv_tot_next)
print("Expected result = 694")

assert abs(p_lv_tot_next - 694) < 1e-9
print("Euler test passed!")


# Test of the complete simulation
model.simulate(t_f=3600, dt=60)

print("Number of time points =", len(model.t))
print("Number of viral titer values =", len(model.p_lv_tot))
print("Initial viral titer =", model.p_lv_tot[0])
print("Final viral titer =", model.p_lv_tot[-1])

assert len(model.t) == len(model.p_lv_tot)
assert model.p_lv_tot[0] == 0.0
assert model.p_lv_tot[-1] > 0

print("Complete simulation test passed!")


import matplotlib.pyplot as plt

plt.plot(model.t / (24 * 3600), model.p_lv_tot)
plt.xlabel("Time (days)")
plt.ylabel("Total lentiviral titer")
plt.title("Total Lentiviral Production (g2)")
plt.grid(True)
plt.show()
