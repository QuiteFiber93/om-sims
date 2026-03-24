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
        self.R = gravity.R
        
    def __repr__(self):
        return f"Celestial Body (Name, ID): {self.name, self.id}"
    
    @classmethod
    def from_spice(cls, name: str, frame: str = None, gravity: GravityModel = None):
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
        
        # Retrieving IDs loaded in SPICE to ensure no conflicting
        try:
            loaded_ids = _get_loaded_spk_bodies()
            
        except Exception:
            loaded_ids = set()
    
        # a custom ID is being requested
        # need to ensure there is no conflicting IDs
        if id is not None:
            
            if id in loaded_ids:
                raise ValueError(f"Requested id {id} already exists in a loaded SPK kernel."
                    "If you are modeling an existing spacecraft, use "
                    "Spacecraft.from_naif() instead. Otherwise, choose a "
                    "different id or unload the conflicting kernel.")
            
            if id > Spacecraft.id:
                self.id = id
                raise warnings.warn(f"Provided id: {id} is behind the class id count: "
                                    f"Spacecraft.id = {Spacecraft.id} indicating an"
                                    "id overlap between two Spacecraft. Consider another "
                                    "id because this may cause unintended behavior. ")
                
            elif id == Spacecraft.id:
                self.id = id
                Spacecraft.id = Spacecraft.id - 1
                
            elif id < Spacecraft.id:
                self.id = id
                Spacecraft.id = id - 1
                
        # If no id is provided to the spacecraft, then the class id is used and iterated
        # Follows naif convention that negative integers are spacecraft
        else:
            # Checks to make sure the auto-assigned id is not in loaded ids either
            # If auto-assigned id is in loaded id, increment until id is clear
            while Spacecraft.id in loaded_ids:
                Spacecraft.id -= 1
                
                # This is just to prevent issues with infinite loops in runtime
                if Spacecraft.id < -100000:
                    break
                
            self.id = Spacecraft.id
            Spacecraft.id = Spacecraft.id - 1
            
        # If no frame is provided, uses default frame of central body
        self.central_body = central_body
        self.frame = frame if frame else central_body.frame
        
        # Clock bias for communications
        self.clock_bias = clock_bias
        
    @classmethod
    def from_spice(
        cls,
        name: str,
        mass: float,
        central_body: CelestialBody,
        frame: str = None,
        id: int = None,
        clock_bias: float = 0
    ):
        
        name = name.upper()
        
        if id is None:
            sc_id, id_found = spice.bodn2c(name)

            if not id_found:
                raise ValueError(f"Could not resolve '{name}' to a NAIF body ID.")
            
            id = sc_id
            
        # Using the __new__() to bypass the SPICE ID check of the __init__
        # This does mean each of the object properties are set individually
        sc = object.__new__(cls)
        sc.name = name
        sc.mass = mass
        sc.id = id
        sc.central_body = central_body
        sc.frame = frame if frame else central_body.frame
        sc.clock_bias = clock_bias
        
        return sc
        
    # Helper function writes a Spacecraft's trajectory to a Type 9 SPK
    def write_to_kernel(self, 
                        et: np.ndarray | list[np.ndarray], 
                        traj: np.ndarray | list[np.ndarray], 
                        filepath: str = None, 
                        degree: int = 7, 
                        segment_id: str = None
                        ):
        
        n = len(et)
        
        if n < 2:
            raise ValueError(f"At least two epochs are needed to write to Kernel. Recieved {n} epochs.")
        
        if traj.shape != (n, 6):
            raise ValueError(f"Trajectory must have shape ({n}, 6). traj has shape {traj.shape}.")
        
        et_strictly_increasing = all(i < j for i,j in zip(et[:-1], et[1:]))
        if not et_strictly_increasing:
            raise ValueError("Epochs of et must be strictly increasing.")
        
        if degree < 1 or degree > 27:
            raise ValueError("SPK Type 9 requires degree to be between 1 and 27.")
        
        if degree % 2 == 0:
            raise ValueError("SPK Type 9 requires degree to be odd.")
        
        if degree >= n:
            raise ValueError("Interpolation requires degree to be less than number of epochs {n}.")
        
        if filepath is None:
            filepath = self.name.replace(" ", "_") + f"_{self.id}.bsp"
        
        if segment_id is None:
            segment_id = f"SPK_{self.name}"
        # Why is this next bit here?
        # Won't it throw an error?
        # segment_id = segment_id[:40]
        
        handle = spice.spkopn(filepath, f"SPK for {self.name}", 0)
        
        try:
            spice.spkw09(
                handle,
                self.id,
                self.central_body.id,
                self.frame,
                et[0],
                et[-1],
                segment_id,
                degree,
                n,
                traj.tolist(),
                et.tolist()
            )
            
        except Exception:
            spice.spkcls(handle)
            raise
            
        spice.spkcls(handle)