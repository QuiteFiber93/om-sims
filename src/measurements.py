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
    
    def v(self):
        return 0

class Range(Measurement):
    """Direct measurement of instantaneous range from observer.
    """
    def __init__(self, statedef: StateDefinition, bias, sigma):
        self.bias = bias
        self.sigma = sigma
        
        if not statedef.has('position'):
            raise ValueError("Position indices not included in StateDefinition: statedef.")
        
        self.pos_idx = statedef['position']
        
    def h(self, et: float, state: np.ndarray):
        """Instantaneous range from observer to target. Typically frame invariant, but assumes topological ENU frame.

        Args:
            et (float): ephemeris time of observation
            state (np.ndarray): state of target relative to observer at time of observation.

        Returns:
            np.ndarray: array of observations
        """
        r = state[self.pos_idx]
        return np.linalg.norm(r, axis = 0)
        
    def H(self, et: float, state: np.ndarray):
        
        r = state[self.pos_idx]
        h_jacobian = np.zeros((1, state.shape[0]))
        h_jacobian[0, self.pos_idx] = r / np.linalg.norm(r)
        
        return h_jacobian
    
    def additive_noise(self, n: int, rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()
        
        return rng.normal(self.bias, self.sigma, n)

class RangeRate(Measurement):
    """Direct measurement of radial component of velocity (range rate)
    """
    def __init__(self, statedef: StateDefinition, bias, sigma):
        self.bias = bias
        self.sigma = sigma
        
        if not statedef.has('position'):
            raise ValueError("Position indices not included in StateDefinition: statedef.")
        
        if not statedef.has('velocity'):
            raise ValueError('Velocity indices not included in StateDefinition: statedef.')
        
        self.pos_idx = statedef['position']
        self.vel_idx = statedef['velocity']
        
    def h(self, et: float, state: np.ndarray):
        r = state[self.pos_idx]
        v = state[self.vel_idx]
        
        return  r @ v / np.linalg.norm(r)
        
    def H(self, et: float, state: np.ndarray):
        r = state[self.pos_idx]
        v = state[self.vel_idx]
        rho = np.linalg.norm(r)
        
        h_jacobian = np.zeros((1, state.size))
        h_jacobian[0, self.pos_idx] = v/rho - (r @ v) * r / rho**3
        h_jacobian[0, self.vel_idx] = r / rho
        
        return h_jacobian

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
        self.measurements = measurements
        
    @property
    def size(self) -> int:
        return len(self.measurements)
    
    # not all measurements will need the station position (thinking about attitude)
    # so idk what to do about that yet
    def h(self, et: float | np.ndarray, state: np.ndarray, station_pos: np.ndarray = None) -> np.ndarray:
        # Checking if et is a list of times, needed for shape of measurements return
        ets = np.atleast_1d(et)
        y = np.zeros((self.size))
        
        for i, measurement in enumerate(self.measurements):
            for t in range(ets.size):
                state_t = state if state.ndim == 1 else state[:, t]
                if station_position is not None:            
                    station_position = station_pos if station_pos.ndim == 1 else station_pos[:, t]
                y[i, t] = measurement.h(ets[t], state_t, station_position)
                
        return np.squeeze(y)
    
    def H(self, et: float | np.ndarray, state: np.ndarray, station_pos: np.ndarray = None, step: float = 1E-16) -> np.ndarray:
        ets = np.atleast_1d(et)
        
        H = np.zeros((self.size, state.size, ets.size))
        
        # loop through measurements
        # check to see if H is defined
        # if H is defined, then calculate Jacobian analytically
        # if H is not defined, then calculate Jacobian with finite difference
        
        pass