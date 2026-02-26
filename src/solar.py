import numpy as np

from src.force import Perturbation

class SolarRadiationPressure(Perturbation):
    def __init__(self):
        pass
    
    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray, area: float):
        pass