import numpy as np
from scipy.optimize import newton as scipy_newton
from src.numerical_methods.rootfinder import newton
from src.checkingkwargs import invalidArgs, checkkwargs
from src.constants import mu_E

def circular_velocity(mu: float, r: float | np.ndarray) -> float:
    return np.sqrt(mu / np.linalg.norm(r))

def escape_velocity(mu: float, r: float | np.ndarray) -> float:
    return np.sqrt(2 * mu / np.linalg.norm(r))

def flight_path_angle(r: np.ndarray = None, v: np.ndarray = None, **kwargs)-> float:
    """Calculates Flight Path Angle

    Args:
        r (np.ndarray, optional): Position vector. Defaults to None.
        v (np.ndarray, optional): Velocity vector. Defaults to None.
        e (float, optional): eccentricity
        ta (float, optional): True anomaly
    Raises:
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    allowable_keywords = ['ta', 'e']
    checkkwargs(allowable_keywords, kwargs)
    
    if (r is not None) and (v is not None):
        rnorm = np.linalg.norm(r)
        
        vr = np.dot(r, v)/rnorm
        vn = np.linalg.norm( v - vr*r/rnorm )
        return np.arctan(vr / vn)
    
    elif set(['e', 'ta']) <= kwargs.keys(): 
        e = kwargs['e']
        ta = kwargs['ta']
        return np.arctan(e*np.sin(ta) / (1 + e*np.cos(ta)))
    
    else:
        raise invalidArgs

def semimajor_axis(r: float | np.ndarray = None, v: float | np.ndarray = None, mu: float = None, **kwargs) -> float:
    """Calculates the semimajor axis of an orbit. Valid kwarg combos ->
        
        r, v, mu,
        
        rp, ra
        
        rp, e
        
        ra, e
        
        mu, period
        
    Args:
        r (float | np.ndarray, optional): Position vector or magnitude. Defaults to None.
        v (float | np.ndarray, optional): Velocity vector or magnitude. Defaults to None.
        mu (float, optional): Gravitational Parameter. Defaults to None.
        ra (float, optional): Apoapse radius
        rp (float, optional): Periapse radius
        period (float, optional): Period of orbit
        e (float, optional): Eccentricity
        

    Returns:
        float: Semimajor axis of orbit. Ellipse if a > 0. Parabola if a == inf. Hyperbolic if a < 0.
    """
    
    acceptable_keywords = ['rp', 'ra', 'e']
    checkkwargs(acceptable_keywords, kwargs)
    
    if (r is not None) and (v is not None) and (mu is not None):
        
        try: 
            a = 1/(2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / mu)
            
        # the above will result in division by 0 if the orbit is parabolic -> a = infty
        except: 
            ZeroDivisionError
            a = np.inf
            
    elif set(['rp', 'ra']) <= kwargs.keys():
        
        rp = kwargs['rp']
        ra = kwargs['ra']
        return (rp + ra) / 2
    
    elif set(['rp, e']) <= kwargs.keys():
        rp = kwargs['rp']
        e = kwargs['e']
        return rp / (1 - e)
    
    elif set(['ra', 'e']) <= kwargs.keys():
        ra = kwargs['ra']
        e = kwargs['e']
        return ra / (1 + e)
    
    elif mu and set(['period']) <= kwargs.keys():
        T = kwargs['period']
        return (T**2 / (4*np.pi**2) * mu)**(1/3)
        
    else:
        raise invalidArgs
    
    return a

def orbit_energy(mu, r: float | np.ndarray = None, v: float | np.ndarray = None, **kwargs) -> float:
    """Calculates Specific Orbital Energy of an orbit. Valid arg combos -> 
    
    r, v
    
    a
    
    h, e

    Args:
        mu (float): Gravitational Parameter of the attracting body
        r (float | np.ndarray): Position vector or magnitude. 
        v (float | np.ndarray): Velocity vector or magnitude. 
        a (float): semimajor axis. 'a' is only used if both r and v are not provided. 
        
    Raises:
        TypeError: Raises a type error if energy can not be calculated from provided arguments

    Returns:
        float: The specific orbital energy
    """
    
    # Checking kwargs
    acceptable_keywords = ['a', 'h', 'e']
    checkkwargs(acceptable_keywords, kwargs)
    
    if r and v:
        print(r, v)
        return np.linalg.norm(v)**2 / 2 - mu / np.linalg.norm(r)
    
    elif set(['a']) <= kwargs.keys():
        a = kwargs['a']
        
        # Lets user know that r or v (but not both) are provided but not used
        if r: print('Argument \'r\' provided but not \'v\'. Ignored in favor of \'a\'.')
        elif v: print('Argument \'v\' provided but not \'r\'. Ignored in favor of \'a\'.')
        
        return -mu / (2*a)
    
    elif set(['h', 'e']) <= kwargs.keys() :
        h = kwargs['h']
        e = kwargs['e']
        return -mu**2 / (2 *np.linalg.norm(h) ) * (1 - np.linalg.norm(e)**2)
    
    else:
        raise invalidArgs

def radial_mag(ta: float = None, a: float = None, e: float = None, **kwargs) -> float:
    """Calculates magnitude of radial vector

    Args:
        ta (float, optional): True anomaly. Defaults to None
        a (float, optional): Semimajor axis. Defaults to None.
        e (float, optional): Eccentricity. Defaults to None.
        h (float, optional): specific angular momentum.
        mu (float, optional): gravitational parameter.
        E (float, optional): Eccentric Anomaly

    Raises:
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    # Checking kwargs
    acceptable_keywords = ['h', 'mu', 'E']
    checkkwargs(acceptable_keywords, kwargs)
    
    if a and e and ta:
        return a*(1-e**2) / (1 + e*np.cos(ta))
    
    elif ta and set(['h', 'mu']) <= kwargs.keys():
        h = kwargs['h']
        mu = kwargs['mu']
        return h**2 / (mu*(1 + e*np.cos(ta)))
    
    elif a and e and (set(['E']) <= kwargs.keys()):
        E = kwargs['E']
        return a * (1 - e*np.cos(E))
    
    else:
        raise invalidArgs

def periapse_radius(e: float, a: float = None, **kwargs) -> float:
    # Checking kwargs
    acceptable_keywords = ['h', 'mu']
    checkkwargs(acceptable_keywords, kwargs)
    
    if a: 
        return a*(1-e)
    
    elif set(['h', 'mu']) <= kwargs.keys():
        h = kwargs['h']
        mu = kwargs['mu']
        return h**2 / (mu*(1+e))
    
    else:
        raise invalidArgs
    
def apoapse_radius(e: float, a: float = None, **kwargs) -> float:
    # Checking kwargs
    acceptable_keywords = ['h', 'mu']
    checkkwargs(acceptable_keywords, kwargs)
    if a: 
        return a*(1+e)
    
    elif set(['h', 'mu']) <= kwargs.keys():
        h = kwargs['h']
        mu = kwargs['mu']
        return h**2 / (mu*(1-e))
    
    else:
        raise invalidArgs
    
# vis-viva equation for calculating velocity
def visviva(mu: float, r: float | np.ndarray , a: float) -> float:
    """Calculates the velocity vector magnitude of an orbit using the vis-viva equation

    Args:
        mu (float): Graviational Parameter of attracting body
        r (float | np.ndarray): position vector or magnitude
        a (float): semimajor axis

    Returns:
        float: velocity magnitude of orbit at given radius
    """
    return np.sqrt(mu * (2/r - 1/a))

def mean_motion(mu: float = None, a: float = None, **kwargs) -> float:
    """Calculates mean motion

    Args:
        mu (float, optional): Gravitational Parameter. Defaults to None.
        a (float, optional): Semimajor axis. Defaults to None.
        period (float, optional): Period of the orbit. Used if mu and a are not provided. 
    Raises:
        ValueError: _description_
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    # Checking kwargs
    acceptable_keywords = ['period']
    checkkwargs(acceptable_keywords, kwargs)
    
    if mu and a:
        
        # Check if semimajor axis is positive -> closed orbit
        if a <= 0: 
            raise ValueError
        (f"Semimajor axis (value = {a}) must be greater than 0")
        
        return np.sqrt(mu / a**3)
    
    # if mu or a is not provided, then use another method
    elif set(['period']) <= kwargs.keys():
        return 2*np.pi / kwargs['period']

    else: 
        raise invalidArgs

def period(mu: float = None, a: float = None, **kwargs) -> float:
    """Calculates period of orbit

    Args:
        mu (float, optional): Gravitational Parameter. Defaults to None.
        a (float, optional): Semimajor axis. Defaults to None.
        n (float, optional): Mean motion. Used if mu and a are not provided.

    Raises:
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    # Checking kwargs
    acceptable_keywords = ['n']
    checkkwargs(acceptable_keywords, kwargs)
    
    if mu and a:
        return 2*np.pi * np.sqrt(a**3 / mu)
    
    elif kwargs.keys() == set(['n']):
        return 2*np.pi / kwargs['n']
    
    else: 
        raise invalidArgs

def asymptotic_ta(e: float) -> float:
    return np.arccos(-1/e)

def turning_angle(e: float) -> float:
    return 2 * np.arcsin(1/e)

def aiming_radius(a: float, e: float) -> float:
    return abs(a) * np.sqrt(e**2 - 1)

def v_infinity(mu: float = None, a: float = None, **kwargs) -> float:
    """_summary_

    Args:
        mu (float, optional): Gravitational Parameter. Defaults to None.
        a (float, optional): Semimajor axis. Defaults to None.
        h (float, optional): Specific angular momentum. Used with e and mu if a is not provided.
        e (float, optional): Eccentricity. Used with h and mu if a is not provided.

    Raises:
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    acceptable_keywords = ['h', 'e']
    checkkwargs(acceptable_keywords, kwargs)
    
    if mu and a:
        return np.sqrt(abs(mu/a))
    
    elif mu and set(['h', 'e']) <= kwargs.keys():
        h = kwargs['h']
        e = kwargs['e']
        return mu/h * np.sqrt(e**2 - 1)
    else:
        raise invalidArgs

def C3(mu: float, a: float) -> float:
    return abs(mu/a)

def angular_momentum(r: np.ndarray = None, v: np.ndarray = None, **kwargs) -> float | np.ndarray:
    """Calculates angular momentum. Valid args ->
    
    r, v
    
    a, e, mu

    Args:
        r (np.ndarray, optional): _description_. Defaults to None.
        v (np.ndarray, optional): _description_. Defaults to None.

    Raises:
        invalidArgs: _description_

    Returns:
        float | np.ndarray: _description_
    """
    # Checking kwargs
    acceptable_keywords = ['a', 'e', 'mu']
    checkkwargs(acceptable_keywords, kwargs)
    
    if r is not None and v is not None :
        return np.cross(r, v)
    
    elif set(['a', 'e', 'mu']) <= kwargs.keys():
        a = kwargs['a']
        e = kwargs['e']
        mu = kwargs['mu']
        return np.sqrt(mu * a * (1 - e**2))
    
    else:
        raise invalidArgs

def eccentricity_vector(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    return np.cross(v, np.cross(r, v))/mu - r/np.linalg.norm(r)

def eccentricity(r: float | np.ndarray = None, v: float | np.ndarray = None, mu: float = None, **kwargs) -> float:
    """Calculates Eccentricity

    Args:
        r (float | np.ndarray, optional): Position. Defaults to None.
        v (float | np.ndarray, optional): Velocity. Defaults to None.
        mu (float, optional): Gravitational Parameter. Defaults to None.
        rp (float, optional): Periapse radius. Used with ra or a if r, v, mu are not provided.
        ra (float, optional): Apoapse radius. Used with rp or a if r, v, mu are not provided.
        a (float, optional): Semimajor axis. Used with rp or ra if r, v, mu are not provided.
    Raises:
        invalidArgs: _description_

    Returns:
        float: _description_
    """
    # Checking kwargs
    acceptable_keywords = ['a', 'ra', 'rp']
    checkkwargs(acceptable_keywords, kwargs)
    
    if (r is not None) and (v is not None) and (mu is not None):
        return np.linalg.norm(np.cross(v, np.cross(r, v))/mu - r/np.linalg.norm(r))
    
    elif set(['rp', 'a']) <= kwargs.keys():
        a = kwargs['a']
        rp = kwargs['rp']
        return 1 - np.linalg.norm(rp)/np.linalg.norm(a)
    
    elif set(['ra', 'a']) <= kwargs.keys():
        a = kwargs['a']
        ra = kwargs['ra']
        return np.linalg.norm(ra)/np.linalg.norm(a) - 1
    
    elif set(['rp', 'ra']) <= kwargs.keys():
        ra = kwargs['ra']
        rp = kwargs['rp']
        return (ra - rp)/ (ra + rp)
    
    else:
        raise invalidArgs

def state2keplerian(r: np.ndarray, 
                    v: np.ndarray, 
                    mu: float, 
                    I = np.array([1, 0, 0]), 
                    J = np.array([0, 1, 0]), 
                    K = np.array([0, 0, 1])
                    ) -> tuple[float]:
    
    # calculing semimajor axis
    a = semimajor_axis(r, v, mu_E)
    
    # calculating specific angular momentum, used a lot
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    
    # calculating eccentricity vector magnitude
    e = np.cross(v, h)/mu - r/np.linalg.norm(r)
    enorm = np.linalg.norm(e)
    
    # calculating inclination
    inclination = np.arccos(np.dot(K, h)/hnorm) 
    
    # calculating nodal vector
    n = np.cross(K, h/hnorm)
        
    nnorm = np.sin(inclination)
    
    # Calculating RAAN, need to deal with the singularities
    # RAAN and AOP are ill defined for inclination = 0
    if np.isclose(inclination, 0, 1E-10):
        raan = np.nan
    else: 
        raan = np.arccos(np.dot(n, I)/nnorm)
        if n[1] < 0: 
            raan = 2*np.pi - raan
    
    # Calculating TA, need to deal with singularities:
    # AOP and TA ill defnied for e = 0
    
    if np.isclose(enorm, 0, 1E-5):
        ta = np.nan
    else:
        
        # clipping to 0 to avoid floating point errors moving the arguemnt out of bounds
        x = np.dot(r, e)/(enorm*np.linalg.norm(r))
        x = max( min(x, 1), -1)
        ta = np.arccos(x)
        if np.dot(r, v) < 0:
            ta = 2*np.pi - ta
            
    # Calculating AOP, need to deal with singularities
    if not (np.isclose(enorm, 0, 1E-10) and np.isclose(inclination, 0, 1E-10)):
        aop = np.arccos(np.dot(n, e) / (nnorm * enorm))
        if np.dot(e, K) < 0:
            aop = 2*np.pi - aop
            
    else: 
        aop = np.nan
    
    
    return a, enorm, inclination, raan, aop, ta

def keplerian2state(a: float, 
                    e: float, 
                    inclination: float, 
                    raan: float, 
                    aop: float, 
                    ta: float,
                    mu: float
                    ) -> tuple[np.ndarray]:
    """Converts Keplerian Orbital Elements to Cartesian State Vector. Not valid for parabolic trajectories.

    Args:
        a (float): semimajor axis
        e (float): eccentricity magnitude
        inclination (float): orbit inclination
        raan (float): Right Ascension of Ascending Node
        aop (float): Argument of Periapse
        ta (float): True Anomaly
        mu (float): Gravitational Parameter

    Returns:
        tuple[np.ndarray]: r, v 
    """
    rnorm = a*(1 - e**2)/(1 + e*np.cos(ta))   
    theta = aop + ta
    r = rnorm * np.array([
        np.cos(raan)*np.cos(theta) - np.sin(raan)*np.sin(theta)*np.cos(inclination),
        np.sin(raan)*np.cos(theta) + np.cos(raan)*np.sin(theta)*np.cos(inclination),
        np.sin(theta)*np.sin(inclination)
    ])
    
    h = angular_momentum(mu = mu, a = a, e = e)
    v = mu/h * np.array([
        - ( np.cos(raan)*( np.sin(theta) + e*np.sin(aop) ) + np.sin(raan)*( np.cos(theta) + e*np.cos(aop) )*np.cos(inclination) ),
        - ( np.sin(raan)*( np.sin(theta) + e*np.sin(aop) ) - np.cos(raan)*( np.cos(theta) +e*np.cos(aop) )*np.cos(inclination) ),
        (np.cos(theta) + e*np.cos(aop))*np.sin(inclination)
    ])
    
    return r,v
        
# convert eccentric anomaly to true anomaly   
def E2ta(e: float, E: float) -> float:
    ta = 2 * np.arctan( np.sqrt( (1+e)/(1-e) )*np.tan(E/2) )

    if ta < 0:
        ta = 2*np.pi + ta
    return ta

# convert true anomaly to eccentric anomaly
def ta2E(e: float, ta: float) -> float:
    E = 2*np.arctan( np.sqrt( (1-e)/(1+e) ) * np.tan(ta/2) )
    
    if E < 0:
        E = 2*np.pi + E
    
    return E

def mean_anomaly(e=None, E = None, **kwargs) -> float:
    # Checking kwargs
    acceptable_keywords = ['n', 't']
    checkkwargs(acceptable_keywords, kwargs)
    
    if e and E:
        return E - e*np.sin(E)
    
    if set(['n', 't']) <= kwargs.keys():
        n = kwargs['n']
        t = kwargs['t']
        return n*t

def kepler_guess(M: float, e: float) -> float:
    u = M + e
    return ( M*(1 - np.sin(u)) + u*np.sin(M) ) / ( 1 + np.sin(M) - np.sin(u) )

def solve_kepler(e: float, M: float = None, tol: float = 1E-10, **kwargs) -> float:
    
    # Checking kwargs
    acceptable_keywords = ['n', 't']
    checkkwargs(acceptable_keywords, kwargs)
    
    if M: 
        def f(E): return E - e*np.sin(E) - M
        def fprime(E): return 1 - e*np.cos(E)
        
    elif set(['n', 't']) <= kwargs.keys():
        n = kwargs['n']
        t = kwargs['t']
        def f(E): return E - e*np.sin(E) - n*t
        
    else: 
        return invalidArgs
    
    Eguess = kepler_guess(M, e)
    E = newton(f, Eguess, fprime = fprime, tol = tol)
    
    return E

def solve_modified_kepler(t: float, t0: float, e: float, n: float, E0: float, tol=1E-10) -> float:
    
    def f(E): return E - e*np.sin(E) - (E0 - e*np.sin(E0)) - n*(t-t0)
    def fprime(E): return 1 - e*np.cos(E)
    
    M0 = E0 - e*np.sin(E0)
    Eguess = n*(t-t0) + M0 + e/2
    return scipy_newton(func = f, fprime = fprime, x0 = Eguess, tol=tol)

def lagrange_time(t: float, t0: float, r0: np.ndarray, v0: np.ndarray, mu: float, **kwargs) -> tuple[np.ndarray]:
    """Calculates state vectors for at time t using initial conditions at time t0. Uses method in Prussing and Conway section 2.4

    Args:
        t (float): currect epoch
        t0 (float): epoch of initial conditions
        r0 (np.ndarray): initial position vector
        v0 (np.ndarray): initial state vector
        mu (float): gravitational parameter
        a (float): semimajor axis. Calculated if not provided
        e (float): eccentricity. Calculated if not provided

    Returns:
        tuple[np.ndarray]: r(t), v(t)
    """
    
    # checking kwargs given
    acceptable_keywords = ['a', 'e']
    checkkwargs(acceptable_keywords, kwargs)
    
    # Calculating needed parameters/elemets
    r0norm = np.linalg.norm(r0)
    
    # semimajor axis, checking if provided or needs to be calculated
    if set(['a']) <= kwargs.keys():
        a = kwargs['a']
        
    else:
        a = semimajor_axis(r0, v0, mu)

    # calculating mean motion
    n = mean_motion(mu, a)
    
    # checking if eccentricity is provided or needs to be calculated
    if set(['e']) <= kwargs.keys():
        e = kwargs['e']
        
    else:
        e = eccentricity(r0, v0, mu)
        
    # Calculating E0
    cosE0 = (1 - r0norm/a)
    sinE0 = np.dot(r0, v0)/(np.sqrt(a*mu))
    E0 = np.arctan2(sinE0, cosE0)

    # Calculating E at epoch
    E = solve_modified_kepler(t, t0, e, n, E0)
    
    # Calculating f and g lagrange coefficients
    f = 1 - a/r0norm * (1 - np.cos(E - E0))
    g = (t - t0) - 1/n * ( (E - E0) - np.sin(E - E0) )
    
    # Finding position vector at epoch
    r = f * r0 + g * v0
    rnorm = np.linalg.norm(r)
    
    # Calculating time derivatives of f and g lagrange coefficients
    fdot = - np.sqrt(mu * a) / (rnorm * r0norm) * np.sin(E - E0)
    gdot = 1 - a/rnorm * (1 - np.cos(E - E0))
    v = fdot*r0 + gdot*v0
    
    return r, v

def lagrange_ta(delta_ta: float, r0: np.ndarray, v0: np.ndarray, mu: float) -> tuple[np.ndarray]:
    """Calculates state vectors for at time t using initial conditions at time t0. Uses method in Curtis Section 2.11.

    Args:
        delta_ta (float): Change in true anomaly
        r0 (np.ndarray): Initial position vector
        v0 (np.ndarray): Initial velocity vector
        mu (float): Gravitational parameter

    Returns:
        tuple[np.ndarray]: _description_
    """
    r0norm = np.linalg.norm(r0)

    vr0 = np.dot(r0, v0)/ r0norm
  
    
    h = np.linalg.norm(angular_momentum(r0, v0))
    
    rnorm = h**2 / (mu + (h**2/r0norm - mu)*np.cos(delta_ta) - h*vr0*np.sin(delta_ta))

    f = 1 - mu*rnorm/h**2 * (1 - np.cos(delta_ta))
    g = rnorm*r0norm/h * np.sin(delta_ta)
    
    r = f*r0 + g*v0
    
    gdot = 1 - mu*r0norm/h**2 * (1 - np.cos(delta_ta))
    fdot = (f*gdot - 1) / g
    
    v = fdot*r0 + gdot*v0
    
    return r, v

# Universal Anomaly Stuff

## Stumpff Functions
def Stumpff_C(z: float | np.ndarray) ->float | np.ndarray[float]:
    return np.array([(np.cosh( np.sqrt(-z_i)) - 1) / (-z_i) if z_i < 0 
             else (1 - np.cos( np.sqrt(z_i)) ) / z_i if z_i > 0 
             else 1/2 if z_i == 0
             else np.nan for z_i in z])

def Stumpff_S(z: float | np.ndarray) ->float | np.ndarray[float]:
    return np.array([( np.sinh( np.sqrt(-z_i) )  - np.sqrt(-z_i)) / np.sqrt(-z_i)**3 if z_i < 0 
            else (np.sqrt(z_i) - np.sin(np.sqrt(z_i))) / np.sqrt(z_i**3) if z_i > 0 
            else 1/6 if z_i == 0
             else np.nan for z_i in z ])

def kepler_universal(X: float, alpha: float, mu: float, t: float, t0: float, sigma0: float, r0: float) -> float:
    """Used to solve keplers equation for universal anomaly.

    Args:
        X (float): Universal anomaly
        alpha (float): Parameter = 1/a
        mu (float): Gravitational parameter
        t (float): time at current epoch
        t0 (float): time at initial epoch
        sigma0 (float): sigma0 = r0 dot v0 / sqrt(mu)
        r0 (float): magnitude of r0 vector

    Returns:
        float
    """
    
    arg = alpha * X**2
    
    return sigma0 * X**2 * Stumpff_C(arg) + (1 - r0*alpha)*X**3 * Stumpff_S(arg) + r0*X - np.sqrt(mu)*(t - t0)

def kepler_prime_universal(X: float, alpha: float, sigma0: float, r0: float) -> float:
    arg = alpha*X**2
    return sigma0*X*( 1 - arg*Stumpff_S(arg) ) + (1 - r0*alpha)*X**2*Stumpff_C(arg) + r0

def kepler_universal_guess():
    pass

def solve_kepler_universal():
    pass


