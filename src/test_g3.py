
from classes_CHG4901 import Bioprocess

model = Bioprocess()

# Test values only (not scientifically calibrated)
model.k_it_toxic = 1e-7
model.u_inf = 1.0
model.k_psi = 1e-5

# Test g3
result = model.g3(x_v=1e6)

print("Calculated g3 =", result)
print("Expected g3 = 1e-6")

assert abs(result - 1e-6) < 1e-12
print("Test g3 passed!")

# Test total cell death rate
k_d_base = model.specific_death_rate(
    c_lac=0,
    c_amm=0,
    c_gln=5
)

k_d_total = k_d_base + result

print("Baseline death rate =", k_d_base)
print("Total death rate =", k_d_total)

assert abs(k_d_total - 3e-6) < 1e-12
print("Total death rate test passed!")


# Compare simulations with and without g3

model_without_g3 = Bioprocess()
model_without_g3.u_inf = 1.0
model_without_g3.k_psi = 1e-5
model_without_g3.k_it_toxic = 0.0

model_without_g3.simulate(t_f=24 * 3600, dt=60)

model_with_g3 = Bioprocess()
model_with_g3.u_inf = 1.0
model_with_g3.k_psi = 1e-5
model_with_g3.k_it_toxic = 1e-7

model_with_g3.simulate(t_f=24 * 3600, dt=60)

print("Viable cells without g3 =", model_without_g3.x_v[-1])
print("Viable cells with g3 =", model_with_g3.x_v[-1])

assert model_with_g3.x_v[-1] < model_without_g3.x_v[-1]

print("Complete g3 simulation test passed!")


import matplotlib.pyplot as plt

plt.plot(
    model_without_g3.t / (24 * 3600),
    model_without_g3.x_v / 1e6,
    label="Without g3"
)

plt.plot(
    model_with_g3.t / (24 * 3600),
    model_with_g3.x_v / 1e6,
    label="With g3"
)

plt.xlabel("Time (days)")
plt.ylabel("Viable Cell Density (10^6 cells/mL)")
plt.title("Effect of Transfection Toxicity on HEK293 Cells")
plt.legend()
plt.grid(True)
plt.show()
