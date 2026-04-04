"""Tests for force.py and forcemodel.py

Tests force base classes and ForceModel dynamics construction.
SPICE calls are mocked so tests run without kernels.

Run with: python -m pytest tests/test_force_and_forcemodel.py -v
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.force import Force, Perturbation
from src.gravity import PointMass, GravityModel
from src.atmosphere import AerodynamicDrag, ConstantAtmosphere
from src.state import StateDefinition, translational_state, translational_attitude_state


# ---- Helpers ---- #

def make_point_mass(name="EARTH", mu=398600.4418, R=6378.137, frame="J2000"):
    return PointMass(name, mu, R, frame)


def make_drag(Cd=2.2, A=1e-6, mass=100.0, rho=1e-12, frame="J2000"):
    atm = ConstantAtmosphere(rho, 300.0)
    return AerodynamicDrag(Cd, A, mass, atm, frame)


class DummyPerturbation(Perturbation):
    """A simple constant-acceleration perturbation for testing the catch-all branch."""
    def __init__(self, acc):
        self.acc = np.array(acc, dtype=float)
    
    def acceleration(self, t, r, v):
        return self.acc


# ============================================================
# force.py tests
# ============================================================

class TestForce:
    
    def test_force_acceleration_raises(self):
        f = Force()
        with pytest.raises(NotImplementedError):
            f.acceleration(0, np.zeros(3), np.zeros(3))
    
    def test_perturbation_returns_none(self):
        """Perturbation base class returns None (not implemented but doesn't raise)."""
        p = Perturbation()
        result = p.acceleration(0, np.zeros(3), np.zeros(3))
        assert result is None
    
    def test_perturbation_is_force(self):
        assert issubclass(Perturbation, Force)


# ============================================================
# ForceModel.__init__ tests
# ============================================================

# Import from updated forcemodel
from src.forcemodel import ForceModel


class TestForceModelInit:
    
    def test_single_force(self):
        grav = make_point_mass()
        fm = ForceModel(grav)
        assert len(fm.forces) == 1
        assert fm.central_body == "EARTH"
        assert fm.frame == "J2000"
    
    def test_multiple_forces(self):
        grav = make_point_mass()
        drag = make_drag()
        fm = ForceModel(grav, drag)
        assert len(fm.forces) == 2
    
    def test_non_perturbation_raises(self):
        with pytest.raises(ValueError, match="not of type Perturbation"):
            ForceModel("not_a_force")
    
    def test_custom_central_body_and_frame(self):
        grav = make_point_mass(name="MARS", mu=42828.0, frame="IAU_MARS")
        fm = ForceModel(grav, central_body="MARS", frame="IAU_MARS")
        assert fm.central_body == "MARS"
        assert fm.frame == "IAU_MARS"
    
    def test_default_state_def(self):
        grav = make_point_mass()
        fm = ForceModel(grav)
        assert fm.state_def.size == 6
        assert fm.state_def.has("position")
        assert fm.state_def.has("velocity")
    
    def test_custom_state_def(self):
        sd = translational_attitude_state()
        grav = make_point_mass()
        fm = ForceModel(grav, state_def=sd)
        assert fm.state_def.size == 13
        assert fm.state_def is sd


# ============================================================
# ForceModel.build_dynamics tests — point mass (no SPICE needed)
# ============================================================

class TestDynamicsPointMass:
    """Tests with a single central-body point mass. No SPICE calls needed
    because the force's name matches central_body and frames match."""
    
    def setup_method(self):
        self.mu = 398600.4418
        self.grav = make_point_mass(mu=self.mu, frame="J2000")
        self.fm = ForceModel(self.grav, central_body="EARTH", frame="J2000")
        self.dynamics = self.fm.build_dynamics()
    
    def test_returns_callable(self):
        assert callable(self.dynamics)
    
    def test_output_shape(self):
        state = np.array([7000.0, 0, 0, 0, 7.546, 0])
        dstate = self.dynamics(0.0, state)
        assert dstate.shape == (6,)
    
    def test_position_derivative_is_velocity(self):
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = self.dynamics(0.0, state)
        np.testing.assert_array_equal(dstate[:3], v)
    
    def test_velocity_derivative_is_gravity(self):
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = self.dynamics(0.0, state)
        
        expected_acc = -self.mu / np.linalg.norm(r)**3 * r
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-12)
    
    def test_circular_orbit_acceleration_magnitude(self):
        """For a circular orbit, |a| = mu/r^2."""
        r_mag = 7000.0
        r = np.array([r_mag, 0, 0])
        v_circ = np.sqrt(self.mu / r_mag)
        v = np.array([0, v_circ, 0])
        state = np.concatenate((r, v))
        dstate = self.dynamics(0.0, state)
        
        acc_mag = np.linalg.norm(dstate[3:6])
        expected = self.mu / r_mag**2
        np.testing.assert_allclose(acc_mag, expected, rtol=1e-12)
    
    def test_acceleration_direction_toward_origin(self):
        """Gravity should point toward the origin."""
        r = np.array([4000.0, 5000.0, 3000.0])
        v = np.array([0, 0, 0])
        state = np.concatenate((r, v))
        dstate = self.dynamics(0.0, state)
        
        acc = dstate[3:6]
        # acc should be antiparallel to r
        cos_angle = np.dot(acc, r) / (np.linalg.norm(acc) * np.linalg.norm(r))
        np.testing.assert_allclose(cos_angle, -1.0, atol=1e-12)


# ============================================================
# ForceModel.build_dynamics — multiple forces (same frame)
# ============================================================

class TestDynamicsMultipleForces:
    """Tests with multiple forces that all share the same frame as the ForceModel,
    so no SPICE frame rotations are needed."""
    
    def test_gravity_plus_constant_perturbation(self):
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        constant_thrust = DummyPerturbation([0.001, 0, 0])
        
        fm = ForceModel(grav, constant_thrust, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        expected_grav = -mu / np.linalg.norm(r)**3 * r
        expected_acc = expected_grav + np.array([0.001, 0, 0])
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-12)
    
    def test_gravity_plus_drag_same_frame(self):
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        drag = make_drag(Cd=2.2, A=1e-6, mass=100.0, rho=1e-12, frame="J2000")
        
        fm = ForceModel(grav, drag, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        # Compute expected values manually
        grav_acc = -mu / np.linalg.norm(r)**3 * r
        drag_acc = drag.acceleration(0.0, r, v)
        expected_acc = grav_acc + drag_acc
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-12)
    
    def test_forces_are_additive(self):
        """Acceleration from two DummyPerturbations should sum."""
        p1 = DummyPerturbation([1.0, 0, 0])
        p2 = DummyPerturbation([0, 2.0, 0])
        
        fm = ForceModel(p1, p2, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        state = np.array([7000.0, 0, 0, 0, 0, 0])
        dstate = dynamics(0.0, state)
        
        np.testing.assert_allclose(dstate[3:6], [1.0, 2.0, 0.0])


# ============================================================
# ForceModel.build_dynamics — third body (mocked SPICE)
# ============================================================

class TestDynamicsThirdBody:
    """Tests the third-body gravity branch where force.name != central_body.
    SPICE is mocked to avoid kernel dependency."""
    
    @patch("src.forcemodel_updated.spice")
    def test_third_body_same_frame(self, mock_spice):
        """Third body in the same frame — no rotation needed."""
        mu_earth = 398600.4418
        mu_moon = 4902.8
        
        earth_grav = make_point_mass(name="EARTH", mu=mu_earth, frame="J2000")
        moon_grav = make_point_mass(name="MOON", mu=mu_moon, frame="J2000")
        
        # Mock: Moon is at [384400, 0, 0] relative to Earth
        moon_pos = np.array([384400.0, 0, 0])
        mock_spice.spkpos.return_value = (moon_pos, 0.0)
        
        fm = ForceModel(earth_grav, moon_grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        # Expected: Earth gravity at r, plus Moon gravity at (r - moon_pos)
        earth_acc = -mu_earth / np.linalg.norm(r)**3 * r
        r_rel_moon = r - moon_pos
        moon_acc = -mu_moon / np.linalg.norm(r_rel_moon)**3 * r_rel_moon
        expected_acc = earth_acc + moon_acc
        
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-10)
        mock_spice.spkpos.assert_called_once_with("MOON", 0.0, "J2000", "NONE", "EARTH")
    
    @patch("src.forcemodel_updated.spice")
    def test_third_body_different_frame(self, mock_spice):
        """Third body in a different frame — rotation should be applied."""
        mu_earth = 398600.4418
        mu_moon = 4902.8
        
        earth_grav = make_point_mass(name="EARTH", mu=mu_earth, frame="J2000")
        moon_grav = make_point_mass(name="MOON", mu=mu_moon, frame="IAU_MOON")
        
        moon_pos = np.array([384400.0, 0, 0])
        mock_spice.spkpos.return_value = (moon_pos, 0.0)
        
        # Use identity rotation so we can verify the path is exercised
        R_identity = np.eye(3)
        mock_spice.pxform.return_value = R_identity
        
        fm = ForceModel(earth_grav, moon_grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        # With identity rotation, result should be same as same-frame case
        earth_acc = -mu_earth / np.linalg.norm(r)**3 * r
        r_rel_moon = r - moon_pos
        moon_acc = -mu_moon / np.linalg.norm(r_rel_moon)**3 * r_rel_moon
        expected_acc = earth_acc + moon_acc
        
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-10)
        mock_spice.pxform.assert_called_once_with("J2000", "IAU_MOON", 0.0)


# ============================================================
# ForceModel.build_dynamics — drag with different frame (mocked SPICE)
# ============================================================

class TestDynamicsDragDifferentFrame:
    
    @patch("src.forcemodel_updated.spice")
    def test_drag_different_frame_identity_rotation(self, mock_spice):
        """Drag in a different frame with identity rotation should give same result."""
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        drag = make_drag(frame="IAU_EARTH")
        
        # Identity transforms
        mock_spice.sxform.return_value = np.eye(6)
        mock_spice.pxform.return_value = np.eye(3)
        
        fm = ForceModel(grav, drag, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        grav_acc = -mu / np.linalg.norm(r)**3 * r
        drag_acc = drag.acceleration(0.0, r, v)
        expected_acc = grav_acc + drag_acc
        
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-10)
        mock_spice.sxform.assert_called_once_with("J2000", "IAU_EARTH", 0.0)
        mock_spice.pxform.assert_called_once_with("IAU_EARTH", "J2000", 0.0)


# ============================================================
# ForceModel.build_dynamics — controller
# ============================================================

class TestDynamicsController:
    
    def test_controller_adds_acceleration(self):
        grav = make_point_mass(frame="J2000")
        
        thrust = np.array([0.0, 0.0, 0.001])
        def controller(t, state, state_def):
            return thrust
        
        fm = ForceModel(grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics(controller=controller)
        
        r = np.array([7000.0, 0, 0])
        v = np.array([0, 7.546, 0])
        state = np.concatenate((r, v))
        dstate = dynamics(0.0, state)
        
        grav_acc = -grav.mu / np.linalg.norm(r)**3 * r
        expected_acc = grav_acc + thrust
        np.testing.assert_allclose(dstate[3:6], expected_acc, rtol=1e-12)
    
    def test_no_controller_by_default(self):
        grav = make_point_mass(frame="J2000")
        fm = ForceModel(grav, central_body="EARTH", frame="J2000")
        
        dynamics_no_ctrl = fm.build_dynamics()
        dynamics_none_ctrl = fm.build_dynamics(controller=None)
        
        state = np.array([7000.0, 0, 0, 0, 7.546, 0])
        
        d1 = dynamics_no_ctrl(0.0, state)
        d2 = dynamics_none_ctrl(0.0, state)
        np.testing.assert_array_equal(d1, d2)
    
    def test_controller_receives_full_state_and_state_def(self):
        """Verify the controller is called with the right arguments."""
        grav = make_point_mass(frame="J2000")
        sd = translational_state()
        fm = ForceModel(grav, central_body="EARTH", frame="J2000", state_def=sd)
        
        captured = {}
        def controller(t, state, state_def):
            captured["t"] = t
            captured["state"] = state.copy()
            captured["state_def"] = state_def
            return np.zeros(3)
        
        dynamics = fm.build_dynamics(controller=controller)
        state = np.array([7000.0, 0, 0, 0, 7.546, 0])
        dynamics(42.0, state)
        
        assert captured["t"] == 42.0
        np.testing.assert_array_equal(captured["state"], state)
        assert captured["state_def"] is sd


# ============================================================
# ForceModel.build_dynamics — StateDefinition integration
# ============================================================

class TestDynamicsStateDefinition:
    
    def test_default_state_def_matches_hardcoded(self):
        """Default 6-state layout should produce same results as explicit."""
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        
        fm_default = ForceModel(grav, central_body="EARTH", frame="J2000")
        fm_explicit = ForceModel(grav, central_body="EARTH", frame="J2000", 
                                  state_def=translational_state())
        
        state = np.array([7000.0, 0, 0, 0, 7.546, 0])
        
        d1 = fm_default.build_dynamics()(0.0, state)
        d2 = fm_explicit.build_dynamics()(0.0, state)
        np.testing.assert_array_equal(d1, d2)
    
    def test_augmented_state_extra_elements_unchanged(self):
        """Extra state elements (like Cd) should remain zero in derivative."""
        sd = StateDefinition([("position", 3), ("velocity", 3), ("Cd", 1)])
        grav = make_point_mass(frame="J2000")
        
        fm = ForceModel(grav, central_body="EARTH", frame="J2000", state_def=sd)
        dynamics = fm.build_dynamics()
        
        state = np.array([7000.0, 0, 0, 0, 7.546, 0, 2.2])
        dstate = dynamics(0.0, state)
        
        assert dstate.shape == (7,)
        # Cd derivative should be zero (no dynamics defined for it)
        assert dstate[6] == 0.0
        # But position and velocity derivatives should still be correct
        np.testing.assert_array_equal(dstate[:3], state[3:6])
    
    def test_attitude_state_quaternion_derivative(self):
        """With attitude blocks, quaternion kinematics should be computed."""
        sd = translational_attitude_state()
        grav = make_point_mass(frame="J2000")
        
        fm = ForceModel(grav, central_body="EARTH", frame="J2000", state_def=sd)
        dynamics = fm.build_dynamics()
        
        # Identity quaternion (scalar-first), spinning about z
        state = np.zeros(13)
        state[sd["position"]] = [7000, 0, 0]
        state[sd["velocity"]] = [0, 7.546, 0]
        state[sd["quaternion"]] = [1, 0, 0, 0]       # identity: [w, x, y, z]
        state[sd["angular_velocity"]] = [0, 0, 0.1]   # spin about z
        
        dstate = dynamics(0.0, state)
        assert dstate.shape == (13,)
        
        # With identity quaternion [1,0,0,0] and omega=[0,0,wz]:
        # dq/dt = 0.5 * q ⊗ [0, omega]
        # dw = 0.5*(-x*wx - y*wy - z*wz) = 0
        # dx = 0.5*( w*wx + y*wz - z*wy) = 0
        # dy = 0.5*( w*wy + z*wx - x*wz) = 0
        # dz = 0.5*( w*wz + x*wy - y*wx) = 0.5*(1*0.1) = 0.05
        dq = dstate[sd["quaternion"]]
        np.testing.assert_allclose(dq, [0, 0, 0, 0.05], atol=1e-15)
    
    def test_attitude_angular_velocity_derivative_is_zero(self):
        """Without torque models, angular velocity derivative should be zero."""
        sd = translational_attitude_state()
        grav = make_point_mass(frame="J2000")
        
        fm = ForceModel(grav, central_body="EARTH", frame="J2000", state_def=sd)
        dynamics = fm.build_dynamics()
        
        state = np.zeros(13)
        state[sd["position"]] = [7000, 0, 0]
        state[sd["velocity"]] = [0, 7.546, 0]
        state[sd["quaternion"]] = [1, 0, 0, 0]
        state[sd["angular_velocity"]] = [0.01, 0.02, 0.03]
        
        dstate = dynamics(0.0, state)
        d_omega = dstate[sd["angular_velocity"]]
        np.testing.assert_array_equal(d_omega, [0, 0, 0])
    
    def test_no_attitude_blocks_skips_quaternion(self):
        """Standard 6-state should not attempt quaternion kinematics."""
        grav = make_point_mass(frame="J2000")
        fm = ForceModel(grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        state = np.array([7000.0, 0, 0, 0, 7.546, 0])
        dstate = dynamics(0.0, state)
        assert dstate.shape == (6,)


# ============================================================
# ForceModel.build_dynamics — conservation / integration sanity
# ============================================================

class TestDynamicsConservation:
    """Sanity checks using short integration steps."""
    
    def test_energy_approximately_conserved(self):
        """A single Euler step should approximately conserve energy for a circular orbit."""
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        fm = ForceModel(grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        r_mag = 7000.0
        v_circ = np.sqrt(mu / r_mag)
        state = np.array([r_mag, 0, 0, 0, v_circ, 0])
        
        # Specific energy at initial state
        def energy(s):
            r = s[:3]
            v = s[3:6]
            return 0.5 * np.dot(v, v) - mu / np.linalg.norm(r)
        
        E0 = energy(state)
        
        # Take a small Euler step
        dt = 1.0  # 1 second
        dstate = dynamics(0.0, state)
        state1 = state + dt * dstate
        E1 = energy(state1)
        
        # Energy should be very close (Euler introduces small error)
        np.testing.assert_allclose(E1, E0, rtol=1e-6)
    
    def test_dynamics_symmetry(self):
        """Dynamics should be symmetric — rotating the state should rotate the output."""
        mu = 398600.4418
        grav = make_point_mass(mu=mu, frame="J2000")
        fm = ForceModel(grav, central_body="EARTH", frame="J2000")
        dynamics = fm.build_dynamics()
        
        # State along x-axis
        state_x = np.array([7000.0, 0, 0, 0, 7.546, 0])
        dstate_x = dynamics(0.0, state_x)
        
        # Same orbit along y-axis
        state_y = np.array([0, 7000.0, 0, -7.546, 0, 0])
        dstate_y = dynamics(0.0, state_y)
        
        # Acceleration magnitudes should match
        acc_x_mag = np.linalg.norm(dstate_x[3:6])
        acc_y_mag = np.linalg.norm(dstate_y[3:6])
        np.testing.assert_allclose(acc_x_mag, acc_y_mag, rtol=1e-12)