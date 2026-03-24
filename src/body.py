import numpy as np
import warnings
import spiceypy as spice

from src.gravity import GravityModel, PointMass

def _get_loaded_spk_bodies() -> set:
    """Returns the set of all body id's currently loaded by SPK kernels

    Returns:
        set: Set of all body id's loaded by spice kernels
    """
    # Variable to contain all of the loaded ids
    body_ids = set()
    
    # Total number of spk kernels loaded in SPICE
    n_spks_loaded = spice.ktotal('SPK')
    
    # Now looping through each and adding listed id's
    for n in range(n_spks_loaded):
        
        # kdata will return the file name of loaded spk along with other info that won't be used
        filepath, _, _, _ = spice.kdata(n, 'spk')
        
        # Attempts to read the file and updates the set accordingly
        try:
            ids = spice.spkobj(filepath)
            body_ids.update(list(ids))
            
        except spice.SpiceyError:
            continue
        
    return body_ids

class Body:
    """Base class for all relevant entities
    """
    def __init__(self):
        pass
    
class CelestialBody(Body):
    """Class for all gravitationally significant entities
    """
    def __init__(self, name: str, id: int, gravity: GravityModel = None, frame: str = "J2000"):
        self.name = name
        self.id = id
        self.gravity = gravity if gravity else PointMass("", 0, 0)
        self.frame = frame
        
    def __repr__(self):
        return f"Celestial Body (Name, ID): {self.name, self.id}"
    
    @classmethod
    def from_naif(cls, name: str, frame: str = None, gravity: GravityModel = None):
        """Builds a CelestailBody object using SPICE kernels

        Args:
            name (str): Name of body as defined in SPICE kernels.
            frame (str, optional): Name of frame as defined in SPICE kernels. If NONE, frame is "IAU_" + name.upper(). Defaults to None.
            gravity (GravityModel, optional): Gravity model to be used for calculations. If NONE, an attempt is made to create a PointMass 
            using information loaded in SPICE. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        # Converts name to upper for consistency
        name = name.upper()
        
        
        # If a frame is not provided, the basic IAU frame will be attempted
        if frame is None:
            frame = f"IAU_{name.upper()}"
        
        # If a gravity model is not loaded already, use a point mass model
        if gravity is None:
            mu = 0
            R = 0
            
            # Searches loaded spice kernels to check for name in list of SPICE IDs
            body_id, found_id = spice.bodn2c(name)
            
            # If the name is not found in the list of IDs, there is an issue and we can't continue
            if not found_id:
                raise ValueError(f"Could not resolve '{name}' to a NAIF body ID. ")
            
            # tries to retrieve GM from spice
            try:
                _, mu = spice.bodvcd(body_id, "GM", 1)[0]
            
            except Exception:
                warnings.warn(f"GM not found in kernel pool for '{name}' (ID {code}). ", UserWarning, stacklevel = 3)   
                
            # tries to retrieve radii from spice
            try:
                _, R = spice.bodvcd(body_id, "RADII", 3)[0]
                
            except Exception:
                warnings.warn(f"RADII not found in kernel pool for '{name}' (ID {code}). ", UserWarning, stacklevel = 3)
            
            gravity = PointMass(name, mu, R, frame)
                    
        return cls(name = name, id = body_id, gravity = gravity, frame = frame)

class Spacecraft(Body):
    """Class for all artificial/non-gravitationally relevant entities
    """
    id = -1
    
    def __init__(self, name: str, mass: float, central_body: CelestialBody, frame: str = None, id: int = None, clock_bias: float = 0):
        self.name = name
        self.mass = mass
        
        # if id is provided, it must be less than or equal to the current class id
        # this is to prevent 
        # TODO: Add logic to check if id is already loaded in spice kernels
        if id is not None:
            
            if id > Spacecraft.id:
                self.id = id
                raise UserWarning(f"Provided id: {id} is behind the class id count: Spacecraft.id = {Spacecraft.id} indicating an"\
                                  "id overlap between two Spacecraft. Consider another id because this may cause unintended behavior. ")
            elif id == Spacecraft.id:
                self.id = id
                Spacecraft.id = Spacecraft.id - 1
            elif id < Spacecraft.id:
                self.id = id
                Spacecraft.id = id - 1
        # If no id is provided to the spacecraft, then the class id is used and iterated
        # Follows naif convention that negative integers are spacecraft
        else:
            id = Spacecraft.id
            self.id = id
            Spacecraft.id = Spacecraft.id - 1
            
        # If no frame is provided, uses default frame of central body
        self.central_body = central_body
        self.frame = frame if frame else central_body.frame
        
        # Clock bias for communications
        self.clock_bias = clock_bias
    
    # Eventual helper function to write to kernel
    def write_to_kernel(self, et: np.ndarray | list[np.ndarray], traj: np.ndarray | list[np.ndarray]):
        
        pass