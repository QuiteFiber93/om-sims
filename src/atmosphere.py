import numpy as np
from src.force import Perturbation
from src.constants import R_E, R_gas, g0, mu_E

class AtmosphereModel:
    """Base class for atmosphere models
    """
    def __init__(self):
        pass
    
class ConstantAtmosphere(AtmosphereModel):
    """Class for atmosphere models which are constant density in time and space
    """
    def __init__(self, density: float, temperature: float):
        self.density = density
        self.temperature = temperature
        
    def __call__(self, t, state):
        return self.density
    
class ExponentialAtmosphere(AtmosphereModel):
    """Class for atmosphere models following exponential atmsophere
    """
    def __init__(self):
        pass
    
    def __call__(self, t: float, pos: np.ndarray) -> float:
        pass
    
class JacchiaRoberts(AtmosphereModel):
    """Class for atmosphere models following Jacchia-Roberts framework
    """
    def __init__(self, F107: float = 150.0, F107avg: float = 150.0, Ht: float = 125, rho0: float = 3E-6, M:float = 0.02897, resolution = 1.0):
        """Model not fully implemented

        Args:
            F107 (float, optional): _description_. Defaults to 150.0.
            F107avg (float, optional): _description_. Defaults to 150.0.
            Ht (float, optional): Atmosphere scale height. Defaults to 125.
            rho0 (float, optional): _description_. Defaults to 3E-6.
            M (float, optional): _description_. Defaults to 0.02897.
            resolution (float, optional): _description_. Defaults to 1.0.
        """
        self.F107 = F107
        self.F107avg = F107avg
        self.T0 = 183 # K
        self.h0 = 90 # km
        self.Ht = Ht
        self.M = M
        self.rho0 = rho0
        self.resolution = resolution
    # Exospheric temperature. Treated as the bounding case as height tends to infinity
    # Acts as the tempertaure forcing
    @property
    def exosphere_temp(self):
        return 379.0 + 3.24 * self.F107avg + 1.3 * (self.F107 - self.F107avg)
    
    @property
    def R_spec(self):
        return R_gas / self.M
    
    def temperature(self, r: float | np.ndarray) -> float | np.ndarray:
        
        # Infinity bound
        Tinf = self.exosphere_temp
        
        # Geometric height
        h = r - R_E
        
        # Calculating geopotential height
        # Required conversion for Bates temperature profile
        z = h * R_E / (R_E + h)
        z0 = self.h0 * R_E / (R_E + self.h0)
        
        T = Tinf - (Tinf - self.T0)*np.exp(-(z - z0) / self.Ht)
        
        T = np.where(h >= self.h0, T, self.T0) 
        
        return float(T) if np.isscalar(r) else T
    
    def _g(self, r: float | np.ndarray) -> float | np.ndarray:
        """Returns gravitational acceleration under spherical gravity

        Args:
            r (float): Radial distance from center of Earth (km)

        Returns:
            float: magnitude of gravity acceleration (km /s^2)
        """
        return mu_E / r**2
    
    def density(self, r: float) -> float:
        """Returns atmospheric density at a given position

        Args:
            r (float): radial position from center of Earth (km)

        Returns:
            float: density (kg/km^3)
        """
        
        
        # Would like a better way to enfore interval on [r0, r]
        r_vals = np.arange(self.h0 + R_E, r, self.resolution) # km
        
        gravity_vals = mu_E / r_vals ** 2 # km/s^2
        T_vals = self.temperature(r_vals) # K
        
        # Should I use Simpson's rule here? 
        # Need to analyze time/memory complexity
        # The improved error scaling sounds fun
        exponential_term = -np.trapz(1E6 * gravity_vals / (self.R_spec * T_vals), r_vals) # should be nondimensional
        return self.rho0 * self.T0 / self.temperature(r) * np.exp(exponential_term)
        
    def __call__(self, t: float, pos: np.ndarray):
        pass
    
class NRLMSISE00(AtmosphereModel):
    """Class for NRLMSISE00 atmosphere model
    """
    def __init__(self):
        pass
    
class AerodynamicDrag(Perturbation):
    """Perturbation class which uses atmosphere models and satellite properties to calculate the aerodynamic drag force on satellites
    """
    def __init__(self, Cd: float, A: float, atmosphere: AtmosphereModel):
        # I don't know if Cd and A should be part of the the Aerodynamic Drag class or parameters of the acceleration
        # self.Cd = None
        # self.A = None 
        self.atmosphere = atmosphere
    
    def acceleration(self, t, r, v):
        rho = self.atmosphere.density()