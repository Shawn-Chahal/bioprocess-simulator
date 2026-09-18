import os

from source.classes import Bioprocess

bioprocess = Bioprocess()
bioprocess.simulate()
bioprocess.visualize(filepath=os.path.join("figures", "Figure_Simulation.png"))
