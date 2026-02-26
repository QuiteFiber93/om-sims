import numpy as np
from src.body import CelestialBody, Spacecraft


# TODO: Try to use spice kernel for ground station and use numpy based methods if kernels not available
#       -> Have an object variable which determines whether or not a spice kernel is loaded for the ground station
#       -> Reference this variable in all functions. If loaded, can use purely spice kernels for transformations
#       -> If not loaded, use spice kernels for transformations to body frame and math for body -> ground station.
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
                 id: float = None,
                 clock_bias: float = 0,
                 use_spice_station = True
                 ):

        self.name = name
        self.frame = frame
        self.lat = lat
        self.lon = lon
        self.altitude = altitude
        self.body = body
        self.el_mask = el_mask
        self.clock_bias = clock_bias
        
        # TODO: FIX ID LOGIC, Make it more similar to the SC id logic
        self.id = int( str(self.body.id) + str(id) )
        self.use_spice_station = use_spice_station
        
    def true_range(self, r: np.ndarray) -> np.ndarray:
        pass
    
    def true_rangerate(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        pass
    
    def measurements(self, et, r, v, observations: list[str]):
        pass