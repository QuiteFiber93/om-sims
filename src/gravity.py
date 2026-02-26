import numpy as np
from typing import Callable
import pyshtools as sh
from src.force import Perturbation

class GravityModel(Perturbation):
    """
    Class to create a gravity model based on physical parameters from a given body
    """
    def __init__(self, name: str, mu: float, R: float, n_max: int, m_max: int, C: np.ndarray, S: np.ndarray):
        """Initializes object for Gravity Model Class

        Args:
            name (str): Name of body
            mu (float): Gravitational Parameter of Body in km^3/s^2
            R (float): Equatorial radius of body
            n_max (int): N
            m_max (int): M
            C (np.ndarray): C coefficients for Legendre Polynomials (2d array)
            S (np.ndarray): S coefficients for Associated Legendre Polynomials (2darray)
        
        Returns:
            None

        Raises:
            ValueError: Raises ValueError if the shape of C and S are not equal to the expected value set by n_max
        """
        self.name = name
        self.mu = mu
        self.R = R
        self.n_max = n_max
        self.m_max = m_max
        
        if C.shape != (n_max + 1, n_max + 1):
            if C.shape[0] != n_max + 1:
                raise ValueError(f'C.shape[0] is not equal to expected value of n_max + 1')
            else: 
                raise ValueError(f'C.shape[1] is not equal to expected value of n_max + 1')
            
        if S.shape != (n_max + 1, n_max + 1):
            if S.shape[0] != n_max + 1:
                raise ValueError(f'S.shape[0] is not equal to expected value of n_max + 1')
            else: 
                raise ValueError(f'S.shape[1] is not equal to expected value of n_max + 1')
        
        self.C = C
        self.S = S
            
    @classmethod
    def from_coefficients(cls, name: str, mu: float, R: float, coefficients: dict) -> 'GravityModel':
        n_max = 0
        m_max = 0
        
        for key in coefficients.keys():
            if key[0] > n_max: n_max = key[0]
            if key[1] > m_max: m_max = key[1]
        
        C = np.zeros((n_max + 1, n_max + 1))
        S = np.zeros((n_max + 1, n_max + 1))
        
        C[0, 0] = 1.0
        
        for (n, m), (C_nm, S_nm) in coefficients.items():
            C[n, m] = C_nm
            S[n ,m] = S_nm
            
        return cls( name = name, mu = mu, R = R, n_max = n_max, m_max = m_max, C = C, S = S )
    
    @classmethod
    def from_file(cls, name: str, mu: float, R: float, n_max: int, m_max: int, file: str) -> 'GravityModel':
        """Creates a GravityModel object from a file containing Stokes Coefficients

        Args:
            name (str): Name of Gravity Model
            mu (float): Gravitational Parameter in km^3 / s^2
            R (float): Equitorial Radius of Planet
            n_max (int): _description_
            m_max (int): _description_
            file (str): File location

        Returns:
            GravityModel: GravityModel Object
        """
        C = np.zeros((n_max+1, n_max+1))
        S = np.zeros((n_max+1, n_max+1))
        
        C[0, 0] = 1.0

        with open(file, 'r') as f:
            for line in f:
                line = line.strip().replace('D', 'E').split()
                
                n = int(line[0])
                m = int(line[1])
                Cnm = float(line[2])
                Snm = float(line[3])
                
                if n > n_max: 
                    break
                
                if m > m_max:
                    continue
                
                C[n, m] = Cnm
                S[n, m] = Snm
        
        return GravityModel(name, mu, R, n_max, m_max, C, S)
    
    def potential(self, t: float, r: np.ndarray, v: np.ndarray = None) -> float:
        """Scalar Gravitational Potential

        Args:
            t (float): time parameter. Not typically used but included for possibly time varying potentials
            r (np.ndarray): position of object in gravity field relative to BCBF origin
            v (np.ndarray, optional): velocity of object in gravity field relative to BCBF coordinate frame. Defaults to None.

        Raises:
            ValueError: Value Error if the shape of position array does not match an expected value (columns are positions)

        Returns:
            float: Float or list of floats representing scalar gravitational potential at each position
        """
        
        # Handling input shape, determines if one or multiple values of r are provided
        positions = np.atleast_2d(r)
        
        # Checks to see if the first dimension is 3 x N
        if r.ndim == 1:
            positions = positions.T
            single_input = True
            
        elif positions.shape[0] == 3:
            single_input = positions.shape[1] == 1
        
        else: raise ValueError(f'Expected r.shape = (3, N) does not match {positions.shape}')

        N = positions.shape[1]
        
        # Conversion to spherical coordinates
        rho = np.linalg.norm(positions, axis = 0)
        lon = np.arctan2(positions[1], positions[0])
        
        sinphi = positions[2] / rho
        gravity_potential = np.zeros(N)
        
        # Looping over each positions
        for k in range(N):
            # Calculating legendre
            P = sh.legendre.PlmBar(self.n_max, sinphi[k])
            for n in range(self.n_max + 1):
                # sum from 0 to n
                for m in range(min(n, self.m_max) + 1):
                    # shtools uses a 1d array. convert 
                    idx = n * (n + 1) // 2 + m
                    gravity_potential[k] += -self.mu / rho[k] * (self.R / rho[k]) ** n * P[idx] * (self.C[n, m]*np.cos(m*lon[k]) + self.S[n, m]*np.sin(m*lon[k]))
            
        # Makes sure return shape correctly corresponds to input shape
        if single_input or N == 1:
            return gravity_potential[0]
        
        return gravity_potential
    
    def acceleration(self, t: float, r: np.ndarray, v: np.ndarray):
        # TODO: Fix floating point accuracy and singularity handling at the poles. 
        
        # Handles input shape
        positions = np.atleast_2d(r)
        
        # Checks to see if the first dimension is 3 x N
        if r.ndim == 1:
            positions = positions.T
            single_input = True
            
        elif positions.shape[0] == 3:
            single_input = positions.shape[1] == 1
        
        else: raise ValueError(f'Expected r.shape = (3, N) does not match {positions.shape}')

        N = positions.shape[1]
        
        # Conversion to spherical coordinates
        rho = np.linalg.norm(positions, axis = 0)
        lon = np.arctan2(positions[1], positions[0])
        
        sinphi = positions[2] / rho
        cosphi = np.sqrt(1 - sinphi**2)
        
        acc_cartesian = np.zeros((3,N))
        for k in range(N):
            
        
            P, dP = sh.legendre.PlmBar_d1(self.n_max, sinphi[k])
            
            dVdrho = 0.0
            dVdphi = 0.0
            dVdpsi = 0.0
        
            for n in range(self.n_max + 1):
                # Values that are used in every iteration of m but change based on n
                nondim_dist = (self.R / rho[k])**n
                idx_mult = n + 1
                for m in range(min(n, self.m_max) + 1):
                    idx = n * (n + 1) // 2 + m
                    dVdrho -= idx_mult * nondim_dist * P[idx] * (self.C[n, m]*np.cos(m*lon[k]) + self.S[n, m]*np.sin(m*lon[k]))
                    dVdphi += nondim_dist * dP[idx] * cosphi[k] * (self.C[n, m]*np.cos(m*lon[k]) + self.S[n, m]*np.sin(m*lon[k]))
                    dVdpsi += nondim_dist * m * P[idx] / cosphi[k] * ( -self.C[n, m] * np.sin(m*lon[k]) + self.S[n, m] * np.cos(m*lon[k]) )
                
            acc = np.array([dVdrho, dVdphi, dVdpsi])
            rotation = np.array([
                [cosphi[k] * np.cos(lon[k]), -sinphi[k]*np.cos(lon[k]), -np.sin(lon[k])],
                [cosphi[k] * np.sin(lon[k]), -sinphi[k]*np.sin(lon[k]), np.cos(lon[k])],
                [sinphi[k], cosphi[k], 0]
            ])
            acc_cartesian[:, k] = self.mu / rho[k]**2 * rotation @ acc
        
        if single_input:
            return acc_cartesian[:, 0]
        
        return acc_cartesian
                
class PointMass(GravityModel):
    def __init__(self, name: str, mu: float, R: float):
        self.name = name
        self.mu = mu
        self.R = R
        self.n_max = 0
        self.m_max = 0
        self.C = np.array([[1]])
        self.S = np.array([[0]])
        
    def potential(self, t: float, r: np.ndarray, v:np.ndarray = None):
        return -self.mu / np.linalg.norm(r)
    
    def acceleration(self, t: float, r: np.ndarray, v:np.ndarray = None):
        return -self.mu/np.linalg.norm(r)**3 * r