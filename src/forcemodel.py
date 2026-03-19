import numpy as np
import spiceypy as spice

from src.force import Perturbation
from src.gravity import GravityModel
from src.atmosphere import AerodynamicDrag
class ForceModel:
    """Class containing all perturbations/forces relevant to dynamics
    """
    def __init__(self, *forces, central_body = 'EARTH', frame = 'J2000'):
        for force in forces:
            if not isinstance(force, Perturbation):
                raise ValueError(f"{force} is not of type Perturbation.")
            
        self.forces = list(forces)
        self.central_body = central_body
        self.frame = frame
    
    def build_dynamics(self):
        """Builds the dynamics based on the list of forces in self.forces and returns a function to be evaluated during integration.
        The dynamics() function has arguments: t (float, the current epoch) and state (np.ndarray, state variables assumed to be expressed in frame of ForceModel.frame).
        """
        def dynamics(t, state):
            """Function to be evaluated during integration

            Args:
                t (float): epoch
                state (np.ndarray): object state at epoch

            Returns:
                np.ndarray: time derivative of state at epoch
            """
            # Parses state variable
            r, v = state[:3], state[3:6]
            
            # Creates acceleration with the same dimensions as v
            acc = np.zeros_like(v)
            
            for force in self.forces:
                
                # Checking to see if a transformation needs to be made to correct position
                
                # Applies acceleration based on type of Perturbation being used
                if isinstance(force, GravityModel):
                    
                    # Behavior changes slightly based on whether a gravity model is a third-body perturbation
                    if force.name == self.central_body:
                        
                        # Relative position is just current position                    
                        relative_position = r

                    # We are now dealing with a third-body effect
                    else:
                        
                        # Relative position is with respect to a different body
                        # So, we need to get the position of third body in this frame
                        # Assumes force.name is the same as the body name in SPICE kernel
                        body_pos = spice.spkpos(force.name, t, self.frame, "NONE", self.central_body)[0]
                                            
                        relative_position = r - body_pos
                    
                    # If the force accepts inputs in a different frame, we need to rotate the current position accordingly
                    # We don't need to rotate velocity because it is not used in gravity model
                    if force.frame == self.frame:
                        acc += force.acceleration(t, relative_position, v)
                        
                    else:
                        R = spice.pxform(self.frame, force.frame, t)
                        relative_position = R @ relative_position
                        
                        # Need to rotate it back to the correct frame
                        acc += R.T @ force.acceleration(t, relative_position, v)
                        
                elif isinstance(force, AerodynamicDrag):
                    # This does not need to check relative position.
                    # Because atmospheric drag should really only matter if we are talking about the primary body
                    # So, I need to either change the logic for handling drag so it is easier to connect to other bodies
                    # Or assume relative position is being passed anyways
                    
                    # Check to see if position and velocity are in the correct frame
                    # if not, need to perform a rotation to the correct frame
                    if force.frame == self.frame:
                        acc += force.acceleration(t, r, v)
                    
                    else:
                        R = spice.sxform(self.frame, force.frame, t)
                        state_rel = R @ state
                        acc += spice.pxform(force.frame, self.frame, t) @ force.acceleration(t, state_rel[:3], state_rel[3:6])
                        
                else:
                    # For now, this is a catch all. The only force types to be impemented are shown above
                    acc += force.acceleration(t, r, v)
            
            return np.concatenate((v, acc))
        
        return dynamics