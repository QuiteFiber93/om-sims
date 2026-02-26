import numpy as np
import spiceypy as spice
from time import perf_counter
from scipy.integrate import solve_ivp
import pyvista as pv
from PIL import Image

from src.gravity import GravityModel, PointMass
from src.orbit import keplerian2state
from src.forcemodel import ForceModel
from src.measurements import generate_NEU_measurements
from src.body import CelestialBody, Spacecraft
from src.orbitview import OrbitView3d

if __name__ == "__main__":
    
    # Loading in metakernel
    mk = "datasets/spice/metakernel.txt"
    print("[INFO]\tLoading metakernel from " + mk)
    start_mk_load = perf_counter()
    spice.furnsh(mk)
    stop_mk_load = perf_counter()
    print(f"[INFO]\tLoaded {spice.ktotal("ALL")} kernels in {(stop_mk_load - start_mk_load):.4f} seconds")
    
    # Start and Stop times for simulation
    start   = "2025-06-21 T10:00:00"
    stop    = "2025-06-22 T10:45:00"
    dt      = 150 //2
    
    # Converting so ephemeris time for use in spice
    et_start    = spice.utc2et(start)
    et_stop     = spice.utc2et(stop)
    N           = int(round((et_stop - et_start) / dt))
    ets         = np.linspace(et_start, et_stop, N)
    
    print("[SIM]\tSimulation Start:\t" + start)
    print("[SIM]\tSimulation Stop:\t" + stop)

    # Setting up relevant forces
    egmfile     = "datasets/gravity/egm2008.txt"
    mu_E        = spice.bodvrd("EARTH", "GM", 1)[1][0]
    R_E         = spice.bodvrd("EARTH", "RADII", 3)[1][0]
    earth_grav  = GravityModel.from_file("EARTH", mu_E, R_E, 8, 8, egmfile)
    EARTH       = CelestialBody("EARTH", 399, earth_grav)
    
    mu_moon     = spice.bodvrd("MOON", "GM", 1)[1][0]
    R_moon      = spice.bodvrd("MOON", "RADII", 3)[1][0]   
    moon_grav   = PointMass("MOON", mu_moon, R_moon)
    MOON        = CelestialBody("MOON", 301, moon_grav)
    
    SC = Spacecraft("SC-001", mass = 1, central_body = EARTH)
    
    # Accumulating forces and creating dynamics
    forcemodel  = ForceModel(EARTH.gravity, MOON.gravity, central_body="EARTH", frame = EARTH.frame)
    dynamics    = forcemodel.build_dynamics()
    
    #initial conditions
    a = 7000
    e = 0.01
    i = np.radians(30)
    raan = np.radians(10)
    aop = np.radians(5)
    ta = np.radians(40)
    initial_state = np.concatenate(keplerian2state(a, e, i, raan, aop, ta, mu_E))
    
    # Propgating state
    sol = solve_ivp(dynamics, [et_start, et_stop], initial_state, "DOP853", ets, rtol = 1E-8, atol = 1E-9)
    
    # Extracting state to retrieve observations
    sc_state = sol.y
    
    ground_station = "DSS-13"
    ground_station_frame = "DSS-13_TOPO"
    elevation_mask = np.radians(10)
    
    # Convert sc state to DSS-14 TOPO frame
    relative_state = np.zeros_like(sc_state)
    for k, et in enumerate(ets):
        R = spice.sxform("J2000", ground_station_frame, et)
        relative_state[:, k] = R @ (sc_state[:, k] - spice.spkezr(ground_station, et, "J2000", "NONE", "EARTH")[0])
        
    # converting relative state to altitude/azimuth frame
   
    alt = np.arctan2(relative_state[2, :], np.linalg.norm(relative_state[:2, :], axis = 0))

    relative_state[:, alt <= elevation_mask] = np.nan
    measurement_type = ["RANGE", "ALT", "AZ", "RANGE RATE"]
    measurements = generate_NEU_measurements(relative_state, *measurement_type)
    
    # Adding noise to measurements
    sigma_range = 1E-2 # accuracy of 10 m
    sigma_rangerate = 1E-4 # accuracy of 1 m/s
    sigma_altaz = np.radians(0.5)

    view = OrbitView3d(
        frame = "J2000",
        off_screen=False,
    )
        
    texture_path = "datasets/textures/earth/bluemarble-8km.jpg"
    view.add_body(
        body = EARTH,
        radius = R_E,
        texture = texture_path,
        resolution=360
    )
    
    view.add_spacecraft(
        SC,
        sc_state[:3, 0],
        size = 100,
        color = 'green'
    )
    
    orbit = pv.lines_from_points(sc_state[:3].T)
    view.plotter.add_mesh(orbit, color = 'red')
    view.plotter.set_background(color = 'black')
    view.plotter.show(auto_close=False)