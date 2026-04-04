import numpy as np
import spiceypy as spice

from src.force import Perturbation
from src.gravity import GravityModel
from src.atmosphere import AerodynamicDrag
from src.state import StateDefinition, translational_state

class ForceModel:
    """Class containing all perturbations/forces relevant to dynamics.
    
    Accepts an optional StateDefinition to support flexible state vector layouts.
    If no StateDefinition is provided, defaults to the standard 6-state 
    [position(3), velocity(3)] layout for backward compatibility.
    """
    def __init__(self, *forces, central_body='EARTH', frame='J2000', state_def: StateDefinition = None):
        for force in forces:
            if not isinstance(force, Perturbation):
                raise ValueError(f"{force} is not of type Perturbation.")
            
        self.forces = list(forces)
        self.central_body = central_body
        self.frame = frame
        
        # Use provided state definition or default to translational
        self.state_def = state_def if state_def is not None else translational_state()
    
    def build_dynamics(self, controller=None):
        """Builds the dynamics based on the list of forces in self.forces and returns 
        a function to be evaluated during integration.
        
        The dynamics() function has arguments: 
            t (float): the current epoch
            state (np.ndarray): state variables, layout defined by self.state_def
            
        Args:
            controller (callable, optional): A function with signature (t, state, state_def) -> np.ndarray
                that returns a control acceleration vector (3,). If None, no control is applied.
                
        Returns:
            callable: dynamics function compatible with scipy.integrate.solve_ivp
        """
        # Cache references for closure performance
        state_def = self.state_def
        forces = self.forces
        central_body = self.central_body
        frame = self.frame
        
        def dynamics(t, state):
            """Function to be evaluated during integration.

            Args:
                t (float): epoch
                state (np.ndarray): object state at epoch

            Returns:
                np.ndarray: time derivative of state at epoch
            """
            # Parse state using StateDefinition
            r = state[state_def["position"]]
            v = state[state_def["velocity"]]
            
            # Initialize full derivative vector
            dstate = np.zeros(state_def.size)
            
            # d(position)/dt = velocity
            dstate[state_def["position"]] = v
            
            # Accumulate acceleration from all forces
            acc = np.zeros(3)
            
            for force in forces:
                
                if isinstance(force, GravityModel):
                    
                    if force.name == central_body:
                        relative_position = r
                    else:
                        body_pos = spice.spkpos(force.name, t, frame, "NONE", central_body)[0]
                        relative_position = r - body_pos
                    
                    if force.frame == frame:
                        acc += force.acceleration(t, relative_position, v)
                    else:
                        R = spice.pxform(frame, force.frame, t)
                        relative_position = R @ relative_position
                        acc += R.T @ force.acceleration(t, relative_position, v)
                        
                elif isinstance(force, AerodynamicDrag):
                    
                    if force.frame == frame:
                        acc += force.acceleration(t, r, v)
                    else:
                        # Need full 6-state transform for drag (velocity matters)
                        sv = np.concatenate((r, v))
                        R6 = spice.sxform(frame, force.frame, t)
                        state_rel = R6 @ sv
                        R3 = spice.pxform(force.frame, frame, t)
                        acc += R3 @ force.acceleration(t, state_rel[:3], state_rel[3:6])
                        
                else:
                    acc += force.acceleration(t, r, v)
            
            # Apply control input if provided
            if controller is not None:
                acc += controller(t, state, state_def)
            
            # d(velocity)/dt = acceleration
            dstate[state_def["velocity"]] = acc
            
            # --- Attitude kinematics and dynamics (if present) ---
            if state_def.has("quaternion") and state_def.has("angular_velocity"):
                q = state[state_def["quaternion"]]
                omega = state[state_def["angular_velocity"]]
                
                # Quaternion kinematics: dq/dt = 0.5 * q ⊗ [0, omega]
                # Scalar-first convention: q = [w, x, y, z] (matches numpy-quaternion)
                w, x, y, z = q[0], q[1], q[2], q[3]
                wx, wy, wz = omega[0], omega[1], omega[2]
                
                dstate[state_def["quaternion"]] = 0.5 * np.array([
                    -x*wx - y*wy - z*wz,
                     w*wx + y*wz - z*wy,
                     w*wy + z*wx - x*wz,
                     w*wz + x*wy - y*wx,
                ])
                
                # Angular velocity dynamics are left as zero for now.
                # When torque models are added, they will fill this in
                # similar to how forces fill acceleration.
                # dstate[state_def["angular_velocity"]] = I_inv @ (torque - omega x (I @ omega))
            
            return dstate
        
        return dynamics