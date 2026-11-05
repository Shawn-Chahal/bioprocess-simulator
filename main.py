import os

from src.classes import Bioprocess

if __name__ == "__main__":
    model = Bioprocess()
    for moi in (0.01, 0.1, 1.0, 10.00):  # etc.
        model.simulate(dt=60, t_f=7 * 24 * 3600, moi=moi, toi=48 * 3600)
        model.visualize(os.path.join("figures", f"Figure_Simulation_MOI_{moi:05.2f}.png"))
