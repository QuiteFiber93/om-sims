import numpy as np
from src.body import CelestialBody, Spacecraft
from src.measurements import Measurement, MeasurementModel

class GroundStation:
    """Class containing useful behaviors for ground station calculations
    """
    def __init__(self, 
                 name: str, 
                 frame: str, 
                 lat: float, 
                 lon: float, 
                 alt: float, 
                 el_mask: float, 
                 observations: MeasurementModel, 
                 id: int = None, 
                 clock_bias: float = 0.0,
                 ):
        pass