import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

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
    """Class for atmosphere models following Jacchia-Roberts framework. This assumes a single molecule atmosphere model.
    """
    def __init__(self, F107: float = 150.0, F107avg: float = 150.0, Ht: float = 125, rho0: float = 3E-6, M:float = 0.02897, resolution = 1.0, h_max = 1000):
        """Model not fully implemented

        Args:
            F107 (float, optional): _description_. Defaults to 150.0.
            F107avg (float, optional): _description_. Defaults to 150.0.
            Ht (float, optional): Atmosphere scale height. Defaults to 125.
            rho0 (float, optional): _description_. Defaults to 3E-6.
            M (float, optional): _description_. Defaults to 0.02897.
            resolution (float, optional): _description_. Defaults to 1.0.
            h_max (float, optional): Maximum altitude for density table (km). Defaults to 1000.0.
        """
        self.F107 = F107
        self.F107avg = F107avg
        self.T0 = 183 # K
        self.h0 = 90 # km
        self.Ht = Ht
        self.M = M
        self.rho0 = rho0
        self.resolution = resolution
        self.h_max = h_max
        
        # Build the interpolation table at construction
        self.build_table(resolution, h_max)
        
    # Exospheric temperature. Treated as the bounding case as height tends to infinity
    # Acts as the tempertaure forcing
    @property
    def exosphere_temp(self):
        return 379.0 + 3.24 * self.F107avg + 1.3 * (self.F107 - self.F107avg)
    
    @property
    def R_spec(self):
        return R_gas / self.M
    
    def build_table(self, resolution: float = None, h_max: float = None) -> None:
        """Precomputes a density lookup table using numerical quadrature, then
        builds a cubic interpolator over log(density) for fast evaluation.
        
        Can be called again to rebuild the table with different parameters,
        e.g. after changing F107, resolution, or altitude bounds.
 
        Args:
            resolution (float, optional): Grid spacing in km. Uses self.resolution if not provided.
            h_max (float, optional): Maximum altitude in km. Uses self.h_max if not provided.
        """
        if resolution is not None:
            self.resolution = resolution
        if h_max is not None:
            self.h_max = h_max
        
        r_min = self.h0 + R_E
        r_max = self.h_max + R_E
        
        # Radial grid for the table
        r_table = np.arange(r_min, r_max + self.resolution, self.resolution)
        
        # Precompute temperature and gravity on the full grid once
        T_vals = self.temperature(r_table)
        g_vals = mu_E / r_table ** 2
        
        # Integrand: g / (R_spec * T), with 1E6 unit conversion factor
        integrand = 1E6 * g_vals / (self.R_spec * T_vals)
        
        # Cumulative trapezoidal integration from r_min outward
        # cumulative_trapezoid gives N-1 values; prepend 0 for the base altitude
        cummulative_integral = cumulative_trapezoid(integrand, r_table, initial=0.0)
        
        # Density at each grid point
        rho_table = self.rho0 * self.T0 / T_vals * np.exp(-cummulative_integral)
        
        # Interpolate in log-space for better accuracy across orders of magnitude
        self._log_rho_interp = interp1d(
            r_table, np.log(rho_table),
            kind='cubic',
            bounds_error=False,
            fill_value=(np.log(rho_table[0]), np.log(rho_table[-1]))
        )
        
        self._r_table_bounds = (r_table[0], r_table[-1])
    
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
        
        if r <= self.h0 + R_E or r <= self._r_table_bounds[0]:
            return self.rho0
        
        return float(np.exp(self._log_rho_interp(r)))
        
    def __call__(self, t: float, pos: np.ndarray):
        r = np.linalg.norm(pos)
        
        return self.density(r)
    
class NRLMSISE00(AtmosphereModel):
    """Class for NRLMSISE00 atmosphere model
    """
    def __init__(self):
        pass
    
class AerodynamicDrag(Perturbation):
    """Perturbation class which uses atmosphere models and satellite properties to calculate the aerodynamic drag force on satellites
    """
    def __init__(self, Cd: float, A: float, mass: float, atmosphere: AtmosphereModel, frame: str = 'IAU_EARTH'):
        # I don't know if Cd and A should be part of the the Aerodynamic Drag class or parameters of the acceleration
        self.Cd = Cd
        self.A = A
        self.mass = mass
        self.atmosphere = atmosphere
        self.frame = frame
    
    def acceleration(self, t, r, v):
        rho = self.atmosphere.density()