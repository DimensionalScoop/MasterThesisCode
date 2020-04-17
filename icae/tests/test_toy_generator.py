import icae.toy.waveformMC as toy
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy import stats

def test_valid():
    distances_to_DOM = np.linspace(10, 160,num=10) #m
    angles_to_DOM = np.deg2rad(np.linspace(0, 180,num=10))
    mean_photons, std_photons = 1000, 300
    photons_per_event = stats.norm(loc = mean_photons, scale = std_photons)

    gen = toy.Generator(distances_to_DOM, angles_to_DOM, photons_per_event)

    gen.generate_valid(100)
    
if __name__ == "__main__":
    test_valid()