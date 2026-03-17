import numpy as np

from src.force import Perturbation

class SolarRadiationPressure(Perturbation):
    """Class for force due to solar radiation pressue
    """
    def __init__(self):
        pass
    
    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray, area: float):
        pass