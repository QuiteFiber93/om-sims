from src.gravity import GravityModel, PointMass
import numpy as np
class Body:
    """Base class for all relevant entities
    """
    def __init__(self):
        pass
    
class CelestialBody(Body):
    """Class for all gravitationally relevant entities
    """
    def __init__(self, name: str, id: int, gravity: GravityModel = None, frame: str = "J2000"):
        self.name = name
        self.id = id
        self.gravity = gravity if gravity else PointMass("", 0, 0)
        self.frame = frame
        
    def __repr__(self):
        return f"Celestial Body (Name, ID): {self.name, self.id}"

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
        if id:
            
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