import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from nrlmsise00 import msise_flat
import spiceypy as spice

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
    def __init__(self, rho0: float = 3.725E-6, h0: float = 90.0, H: float = 8.5):
        
        self.rho0 = rho0
        self.h0 = h0
        self.H = H
        
    def density(self, r: float):
        h = r - R_E
        
        if h < self.h0:
            return self.rho0
        
        return self.rho0 * np.exp(-(h - self.h0) / self.H)
    
    def __call__(self, t: float, pos: np.ndarray) -> float:
        
        r = np.linalg.norm(pos)
        return self.density(r)
    
class ExponentialAtmosphere(AtmosphereModel):
    """Class for atmosphere models following exponential atmosphere. 
    
    Note: rho0 is stored in SI (kg/m^3), but density output is converted
    to kg/km^3 for compatibility with the km-based force model.
    """
    def __init__(self, rho0: float = 3.725E-6, h0: float = 90.0, H: float = 8.5):
        """Initializes Exponential Atmosphere Model
 
        Args:
            rho0 (float): Reference density at h0 (kg/m^3)
            h0 (float): Reference altitude (km)
            H (float): Scale height (km)
        """
        self.rho0 = rho0
        self.h0 = h0
        self.H = H
        
    def density(self, r: float):
        """Returns atmospheric density at a given radial distance.
        
        Internally computed in kg/m^3, converted to kg/km^3 on output
        for compatibility with the km-based force model.
 
        Args:
            r (float): Radial distance from center of Earth (km)
 
        Returns:
            float: Density (kg/km^3)
        """
        
        h = r - R_E
        
        if h < self.h0:
            return self.rho0 * 1E9
        
        return self.rho0 * np.exp(-(h - self.h0) / self.H) * 1E9
    
    def __call__(self, t: float, pos: np.ndarray) -> float:
        
        r = np.linalg.norm(pos)
        return self.density(r)
    
class JacchiaRoberts(AtmosphereModel):
    """Class for atmosphere models following Jacchia-Roberts framework. This assumes a single molecule atmosphere model.
    
    Note: rho0 and M are stored in SI (kg/m^3 and kg/mol), but density output
    is converted to kg/km^3 for compatibility with the km-based force model.
    """
    def __init__(self, F107: float = 150.0, F107avg: float = 150.0, Ht: float = 125, rho0: float = 3E-6, M:float = 0.02897, resolution = 1.0, h_max = 1000):
        """Initializes JacchiaRoberts Atmosphere Model
 
        Args:
            F107 (float, optional): Daily 10.7 cm solar radio flux (SFU). Defaults to 150.0.
            F107avg (float, optional): 81-day average of F10.7 (SFU). Defaults to 150.0.
            Ht (float, optional): Atmosphere scale height (km). Defaults to 125.
            rho0 (float, optional): Reference density at h0 (kg/m^3). Defaults to 3E-6.
            M (float, optional): Mean molecular mass of air (kg/mol). Defaults to 0.02897.
            resolution (float, optional): Grid spacing for density table (km). Defaults to 1.0.
            h_max (float, optional): Maximum altitude for density table (km). Defaults to 1000.
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
    # Acts as the temperature forcing
    @property
    def exosphere_temp(self):
        """Exospheric temperature (K) based on F10.7 solar flux."""
        return 379.0 + 3.24 * self.F107avg + 1.3 * (self.F107 - self.F107avg)
    
    @property
    def R_spec(self):
        """Specific gas constant for air (J/(kg*K) = m^2/(s^2*K))."""
        return R_gas / self.M
    
    def build_table(self, resolution: float = None, h_max: float = None) -> None:
        """Precomputes a density lookup table using numerical quadrature, then
        builds a cubic interpolator over log(density) for fast evaluation.
        
        The integrand g/(R_spec*T) has units of 1/km:
            g: mu_E / r^2 (km/s^2)
            R_spec * T: (m^2/(s^2*K)) * K = m^2/s^2
            g / (R_spec * T): (km/s^2) / (m^2/s^2) = km/m^2 = 1E3/m = 1/km
        
        Integration over r (km) yields a dimensionless exponent.
        
        Can be called again to rebuild the table with different parameters,
        e.g. after changing F107, resolution, or altitude bounds.
 
        Args:
            resolution (float, optional): Grid spacing (km). Uses self.resolution if not provided.
            h_max (float, optional): Maximum altitude (km). Uses self.h_max if not provided.
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
        
        # Integrand: g / (R_spec * T), units: 1/km
        integrand = g_vals / (self.R_spec * T_vals)
        
        # Cumulative trapezoidal integration from r_min outward
        # cumulative_trapezoid gives N-1 values; prepend 0 for the base altitude
        cummulative_integral = cumulative_trapezoid(integrand, r_table, initial=0.0)
        
        # Density at each grid point (kg/m^3, same units as rho0)
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
        """Bates temperature profile.
 
        Args:
            r (float | np.ndarray): Radial distance from center of Earth (km)
 
        Returns:
            float | np.ndarray: Temperature (K)
        """
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
            r (float | np.ndarray): Radial distance from center of Earth (km)
 
        Returns:
            float | np.ndarray: Gravitational acceleration (km/s^2)
        """
        return mu_E / r**2
    
    def density(self, r: float) -> float:
        """Returns atmospheric density at a given radial distance.
        
        Internally computed in kg/m^3, converted to kg/km^3 on output
        for compatibility with the km-based force model.
 
        Args:
            r (float): Radial distance from center of Earth (km)
 
        Returns:
            float: Density (kg/km^3)
        """
        
        if r <= self.h0 + R_E or r <= self._r_table_bounds[0]:
            return self.rho0 * 1E9
        
        return float(np.exp(self._log_rho_interp(r))) * 1E9
        
    def __call__(self, t: float, pos: np.ndarray):
        """Returns atmospheric density at a given position.
 
        Args:
            t (float): Epoch (seconds past reference)
            pos (np.ndarray): Position vector from center of Earth (km)
 
        Returns:
            float: Density (kg/km^3)
        """
        r = np.linalg.norm(pos)
        
        return self.density(r)
    
class NRLMSISE00(AtmosphereModel):
    """Class for NRLMSISE00 atmosphere model
    """
    def __init__(self, F107: float = 150.0, F107avg: float = 150.0, Ap: float = 4.0):
        """

        Args:
            F107 (float, optional): Daily 10.7 cm solar radio flux for the previous day. Defaults to 150.0.
            F107avg (float, optional): 81-day centered average of F10.7. Defaults to 150.0.
            Ap (float, optional): Daily geomagnetic Ap index. Defaults to 4.0.
        """
        self.F107 = F107
        self.F107avg = F107avg
        self.Ap = Ap
    
    def density(self, t: float, pos: np.ndarray) -> float:
        """Returns atmospheric density at a given time and position.
        
        Calls NRLMSISE-00 via msise_flat(), which returns total mass
        density in g/cm^3. This function converts this into kg/km^3 on output.

        Args:
            t (float): ephemeris time
            pos (np.ndarray): Position vector in ECEF frame (km).

        Returns:
            float: Density (kg/m^3)
        """
        dt = spice.et2datetime(t)
        
        # Earth oblateness
        f = 1.0 / 298.257223563
        
        # Getting geodetic coords from spicepy
        lon, lat, altitude = spice.recgeo(pos, R_E, f)
        
        rho = msise_flat(dt, altitude, np.degrees(lat), np.degrees(lon), self.F107avg, self.F107, self.Ap)[5]*1E12
        
        return float(rho)
    
    def __call__(self, t: float, pos: np.ndarray) -> float:
        return self.density(t, pos)
    
class AerodynamicDrag(Perturbation):
    """Perturbation class which uses atmosphere models and satellite properties to calculate the aerodynamic drag force on satellites.
    All inputs and outputs use km-based units for consistency with the force model.
    """
    def __init__(self, Cd: float, A: float, mass: float, atmosphere: AtmosphereModel, frame: str = 'IAU_EARTH'):
        # I don't know if Cd and A should be part of the the Aerodynamic Drag class or parameters of the acceleration
        self.Cd = Cd
        self.A = A
        self.mass = mass
        self.atmosphere = atmosphere
        self.frame = frame
    
    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray):
        
        return -0.5 * self.atmosphere(t, r) * self.Cd * self.A / self.mass * np.linalg.norm(v) * v