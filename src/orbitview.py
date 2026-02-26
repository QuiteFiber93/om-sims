import pyvista as pv
from dataclasses import dataclass
from typing import Callable
import numpy as np
from src.body import CelestialBody, Spacecraft

@dataclass
class CenteringPolicy:
    name: str = "EARTH"
    func: Callable = None
    
class OrbitView3d:
    def __init__(self,
                 frame: str,
                 current_epoch: float = 0,
                 centering_policy: CenteringPolicy = CenteringPolicy("NONE", lambda et: np.zeros(3,)),
                 window_size: tuple[int, int] = (1280, 720),
                 background: str | tuple = "black",
                 show_axes: bool = False,
                 add_lighting: bool = True,
                 off_screen = True,
                 ):
        """Wrapper for a PyVista Plotter Object to include useful behavior

        Args:
            frame (str): Referrence frame for coordinates in plotter
            current_epoch (float, optional): Current time epoch for plotter. Defaults to 0.
            centering_policy (_type_, optional): Function to define the center of plotter coords. Defaults to CenteringPolicy("NONE", lambda et: np.zeros(3,)).
            window_size (tuple[int, int], optional): Size of the view window. Defaults to (1280, 720).
            background (str | tuple, optional): Color of background for plotter. Defaults to "black".
            show_axes (bool, optional): Toggles display of coordinate axes. Defaults to False.
            add_lighting (bool, optional): _description_. Defaults to True.
            off_screen (bool, optional): _description_. Defaults to True.
        """
        self.frame = frame
        self.centering = centering_policy
        self.window_size = window_size
        self.background = background
        self.show_axes = show_axes
        self.off_screen = off_screen
        
        self.plotter = pv.Plotter(window_size = self.window_size, off_screen = self.off_screen)
        self.plotter.set_background(self.background)

        if self.show_axes: self.plotter.show_axes()
        
        if add_lighting: self.plotter.enable_lightkit()
        
        self.current_epoch = current_epoch
        
        self.actors = {}
        self.meshes = {}
        
    def add_body(self, body: CelestialBody, radius: float = None, texture: str = None, color: str = 'blue', resolution: int = None) -> None:
        if body.id in self.actors.keys():
            raise ValueError(f"Error with {body}. A body with ID: {body.id} is already included in the scene actors.")
        
        if not resolution:
            resolution = 60
        
        # Creating sphere mesh
        sphere = pv.Sphere(
            radius = radius,
            theta_resolution = resolution,
            phi_resolution = resolution,
        )
        
        # Apply texture mapping and UV flip if a texture is provided
        tex = None
        if texture:
            # Just make sure that the texture and color dont mix
            color = None
            
            # Changing texture coords/orientation
            tex = pv.read_texture(texture)
            sphere.texture_map_to_sphere(inplace=True, prevent_seam=False)
            uv = sphere.active_texture_coordinates
            uv[:, 1] = 1.0 - uv[:, 1]  # Flip V for correct orientation
            sphere.active_texture_coordinates = uv
        
        # Creating actor
        actor = self.plotter.add_mesh(
            sphere,
            texture = tex,
            color = color,
            smooth_shading = True
        )
        
        self.actors[body.id] = {
            'name' : body.name,
            'body' : body,
            'actor' : actor,
        }
        
    def add_spacecraft(self, sc: Spacecraft, position: np.ndarray = None, size: int = 50, color = 'red') -> None:
        if sc.id in self.actors.keys():
            raise ValueError(f'Error with {sc}. A bdoy with ID: {sc.id} is already included in the scene actors')
        sphere = pv.Sphere(
            radius = size,
            center = position,
            theta_resolution = 20,
            phi_resolution = 20
        )
        
        actor = self.plotter.add_mesh(sphere, color = color)
        self.actors[sc.id] = {
            'name' : sc.name,
            'body' : sc,
            'actor' : actor
        }
    
    def add_trajectory(self):
        pass
    
    def remove_actor(self, actor):
        self.plotter.remove_actor(self.actors[actor.id][actor])
        del self.actors[actor.id]