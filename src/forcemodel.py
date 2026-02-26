from src.gravity import Perturbation, GravityModel
import numpy as np
import spiceypy as spice
class ForceModel:
    def __init__(self, *forces, central_body = 'EARTH', frame = 'J2000'):
        for force in forces:
            if not isinstance(force, Perturbation):
                raise ValueError(f"{force} is not of type Perturbation.")
            
        self.forces = list(forces)
        self.central_body = central_body
        self.frame = frame
    
    def build_dynamics(self):
        def dynamics(t, state):
            # Parses state variable
            r, v = state[:3], state[3:6]
            
            # Creates acceleration with the same dimensions as v
            acc = np.zeros_like(v)
            
            for force in self.forces:
                # Applies acceleration based on type of Perturbation being used
                if isinstance(force, GravityModel):
                    
                    # Behavior changes slightly based on whether a gravity model is a third-body perturbation
                    if force.name == self.central_body:
                        relative_position = r
                        body_pos = np.zeros(3)
                    else:
                        body_pos = spice.spkpos(force.name, t, self.frame, "NONE", self.central_body)[0]
                        relative_position = r - body_pos
                    
                    # Gravity model and required frame changes based on PointMass vs Spherical Harmonics
                    if force.n_max == 0:
                        acc += force.acceleration(t, relative_position, v)
                        
                        if force.name != self.central_body: 
                            acc -= force.mu * body_pos / np.linalg.norm(body_pos)**3
                        
                    else:
                        body_frame = "IAU_" + force.name
                        R = spice.pxform(self.frame, body_frame, t)
                        relative_position = R @ relative_position
                        acc += R.T @ force.acceleration(t, relative_position, v)
                
                else:
                    acc += force.acceleration(t, r, v)
            
            return np.concatenate((v, acc))
        
        return dynamics