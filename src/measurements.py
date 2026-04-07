import numpy as np

class Measurement:
    """Base class for Measurement values such as range, range rate, altitude, and azimuth
    """
    def __init__(self, bias: float | np.array, sigma: float | np.array):
        pass
    
    def h(self):
        pass
    
    def H(self):
        pass
    
    def noise(self):
        return 0

class Range(Measurement):
    pass

class RangeRate(Measurement):
    pass

class PositionAngles(Measurement):
    pass

class MeasurementModel:
    def __init__(self, measurements: list[Measurement]):
        pass