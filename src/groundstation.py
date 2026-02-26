import numpy as np
from src.body import CelestialBody, Spacecraft

class GroundStation:
    """Class containing useful behaviors for ground station calculations
    """
    def __init__(self, 
                 name: str, 
                 frame: str, 
                 lat: float, 
                 lon: float, 
                 altitude: float, 
                 body: CelestialBody, 
                 el_mask: float, 
                 noise: dict, 
                 clock_bias: float = 0
                 ):

        self.name = name
        self.frame = frame
        self.lat = lat
        self.lon = lon
        self.altitude = altitude
        self.body = body
        self.el_mask = el_mask
        self.clock_bias = clock_bias
        
    def oneway_range(self, spacecraft: Spacecraft):
        pass
    
    def twoway_range(self, spacecraft: Spacecraft):
        pass
    
    def range_rate(self, spacecraft: Spacecraft):
        pass
    
    def elevation(self, spacecraft: Spacecraft):
        pass
    
    def azimuth(self, spacecraft: Spacecraft):
        pass
    
    def generate_measurements(self, epoch: float | np.ndarray[float], spacecraft: Spacecraft, types = "ALL", noise: float | np.ndarray = 0) -> np.ndarray:
        # TYPES:
        # OWR -> One Way Range
        # TWR -> Two Way Range
        # RR -> Range Rate
        # EL -> Elevation
        # AZ -> Azimuth
        # ALL: ALL
        # Loops through types of measurements in array to generate measurements for all epochs
        pass