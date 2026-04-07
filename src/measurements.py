import numpy as np

from src.state import StateDefinition

class Measurement:
    """Base class for Measurement values such as range, range rate, altitude, and azimuth
    """
    def __init__(self, statedef: StateDefinition, bias: float | np.array, sigma: float | np.array):
        pass
    
    def h(self):
        pass
    
    def H(self):
        pass
    
    def noise(self):
        return 0

class Range(Measurement):
    def __init__(self, statedef: StateDefinition, bias, sigma):
        self.bias = bias
        self.sigma = sigma
        
        if not statedef.has('position'):
            raise ValueError("Position indices not included in StateDefinition: statedef.")
        
        self.pos_idx = statedef['position']
        
    def h(self, et: float | np.ndarray, state: np.ndarray, station_pos: np.ndarray):
        r = state[self.pos_idx]
        return np.linalg.norm(r - station_pos)
        
    def H(self, et: float | np.ndarray, r: np.ndarray):
        pass

class RangeRate(Measurement):
    def __init__(self, statedef: StateDefinition, bias, sigma):
        self.bias = bias
        self.sigma = sigma
        
        if not statedef.has('position'):
            raise ValueError("Position indices not included in StateDefinition: statedef.")
        
        if not statedef.has('velocity'):
            raise ValueError('Velocity indices not included in StateDefinition: statedef.')
        
        self.pos_idx = statedef['position']
        self.vel_idx = statedef['velocity']
        
    def h(self, et: float | np.ndarray, state: np.ndarray, station_pos: np.ndarray):
        r = state[self.pos_idx]
        v = state[self.vel_idx]
        
        return  r @ v / np.linalg.norm(r - station_pos)

class PositionAngles(Measurement):
    def __init__(self, statedef: StateDefinition, bias, sigma):
        self.bias = bias
        self.sigma = sigma
        
        if not statedef.has('position'):
            raise ValueError("Position indices not included in StateDefinition: statedef.")
        
        self.pos_idx = statedef['position']
        
    def h(self, et: float | np.ndarray, r: np.ndarray, station_pos: np.ndarray):
        pass

class MeasurementModel:
    def __init__(self, measurements: list[Measurement]):
        pass