
from classes_CHG4901 import Bioprocess

model = Bioprocess()

# Test values (not calibrated parameters)
model.f_inf = 0.5
model.u_inf = 1.0
model.k_psi = 1e-5
model.k_p = 0.001
model.k_inact = 0.002

# Test g1
result = model.ddt_p_lv_inf(
    x_v=1e6,
    p_lv_inf=100
)

print("Calculated g1 =", result)
print("Expected g1 = 4.7")

assert abs(result - 4.7) < 1e-9
print("Test g1 passed!")


# Test of Euler's method for g1
dt = 60
p_lv_inf_initial = 100

p_lv_inf_next = max(0, p_lv_inf_initial + result * dt)

print("Euler result =", p_lv_inf_next)
print("Expected result = 382")

assert abs(p_lv_inf_next - 382) < 1e-9
print("Euler g1 test passed!")


# Test of the complete simulation for g1
model.simulate(t_f=3600, dt=60)

print("Number of time points =", len(model.t))
print("Number of infectious viral titer values =", len(model.p_lv_inf))
print("Initial infectious viral titer =", model.p_lv_inf[0])
print("Final infectious viral titer =", model.p_lv_inf[-1])

assert len(model.t) == len(model.p_lv_inf)
assert model.p_lv_inf[0] == 0.0
assert model.p_lv_inf[-1] > 0

print("Complete g1 simulation test passed!")


import matplotlib.pyplot as plt

plt.plot(model.t / (24 * 3600), model.p_lv_inf)
plt.xlabel("Time (days)")
plt.ylabel("Infectious lentiviral titer")
plt.title("Infectious Lentiviral Production (g1)")
plt.grid(True)
plt.show()
