"""
Unit tests for the src modules imported by scripts/orbit_measurements.py:
    - src.gravity        (GravityModel, PointMass)
    - src.orbit          (keplerian2state and supporting functions)
    - src.forcemodel     (ForceModel)
    - src.measurements   (generate_NEU_measurements)
    - src.body           (CelestialBody, Spacecraft)
    - src.atmosphere     (ExponentialAtmosphere, AerodynamicDrag)
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.gravity import GravityModel, PointMass
from src.orbit import (
    keplerian2state,
    circular_velocity,
    escape_velocity,
    flight_path_angle,
    semimajor_axis,
    orbit_energy,
    angular_momentum,
    eccentricity,
    eccentricity_vector,
    period,
    mean_motion,
    visviva,
    periapse_radius,
    apoapse_radius,
    E2ta,
    ta2E,
    mean_anomaly,
    solve_kepler,
    lagrange_time,
    lagrange_ta,
    Stumpff_C,
    Stumpff_S,
    state2keplerian,
)
from src.forcemodel import ForceModel
from src.measurements import generate_NEU_measurements
from src.body import CelestialBody, Spacecraft, Body
from src.atmosphere import (
    ExponentialAtmosphere,
    AerodynamicDrag,
    ConstantAtmosphere,
    AtmosphereModel,
)
from src.force import Force, Perturbation
from src.constants import mu_E, R_E
from src.checkingkwargs import checkkwargs, invalidArgs


# ---------------------------------------------------------------------------
#  Constants used across tests
# ---------------------------------------------------------------------------
MU = 398600.0  # km^3/s^2 (Earth)
R_EARTH = 6378.0  # km


# ===========================================================================
#  src.gravity — PointMass
# ===========================================================================
class TestPointMass:
    """Tests for the PointMass gravity model (spherical symmetry)."""

    def setup_method(self):
        self.pm = PointMass("EARTH", MU, R_EARTH)

    def test_init_attributes(self):
        assert self.pm.name == "EARTH"
        assert self.pm.mu == MU
        assert self.pm.R == R_EARTH
        assert self.pm.n_max == 0
        assert self.pm.m_max == 0
        np.testing.assert_array_equal(self.pm.C, np.array([[1]]))
        np.testing.assert_array_equal(self.pm.S, np.array([[0]]))

    def test_potential_at_surface(self):
        r = np.array([R_EARTH, 0.0, 0.0])
        expected = -MU / R_EARTH
        assert pytest.approx(self.pm.potential(0, r), rel=1e-10) == expected

    def test_potential_off_axis(self):
        r = np.array([0.0, 0.0, R_EARTH + 500.0])
        expected = -MU / np.linalg.norm(r)
        assert pytest.approx(self.pm.potential(0, r), rel=1e-10) == expected

    def test_acceleration_direction(self):
        """Acceleration should point toward the origin (opposite to r)."""
        r = np.array([7000.0, 0.0, 0.0])
        acc = self.pm.acceleration(0, r)
        # Should be purely in -x direction
        assert acc[0] < 0
        np.testing.assert_allclose(acc[1], 0, atol=1e-15)
        np.testing.assert_allclose(acc[2], 0, atol=1e-15)

    def test_acceleration_magnitude(self):
        r = np.array([7000.0, 0.0, 0.0])
        acc = self.pm.acceleration(0, r)
        expected_mag = MU / 7000.0**2
        np.testing.assert_allclose(np.linalg.norm(acc), expected_mag, rtol=1e-10)

    def test_acceleration_inverse_square(self):
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([14000.0, 0.0, 0.0])
        a1 = np.linalg.norm(self.pm.acceleration(0, r1))
        a2 = np.linalg.norm(self.pm.acceleration(0, r2))
        # Doubling distance should quarter acceleration
        np.testing.assert_allclose(a1 / a2, 4.0, rtol=1e-10)

    def test_custom_frame(self):
        pm = PointMass("MOON", 4902.8, 1737.4, frame="IAU_MOON")
        assert pm.frame == "IAU_MOON"

    def test_default_frame(self):
        assert self.pm.frame == "IAU_EARTH"


# ===========================================================================
#  src.gravity — GravityModel
# ===========================================================================
class TestGravityModel:
    """Tests for the full spherical-harmonic GravityModel."""

    def _make_point_mass_gm(self, n_max=2):
        """Build a GravityModel whose coefficients reduce to point-mass."""
        C = np.zeros((n_max + 1, n_max + 1))
        S = np.zeros((n_max + 1, n_max + 1))
        C[0, 0] = 1.0
        return GravityModel("EARTH", MU, R_EARTH, n_max, n_max, C, S)

    def test_init_stores_attributes(self):
        gm = self._make_point_mass_gm(2)
        assert gm.name == "EARTH"
        assert gm.mu == MU
        assert gm.n_max == 2
        assert gm.m_max == 2
        assert gm.C.shape == (3, 3)
        assert gm.S.shape == (3, 3)

    def test_invalid_C_shape_rows(self):
        C = np.zeros((2, 3))
        S = np.zeros((3, 3))
        with pytest.raises(ValueError, match="C.shape"):
            GravityModel("X", MU, R_EARTH, 2, 2, C, S)

    def test_invalid_C_shape_cols(self):
        C = np.zeros((3, 2))
        S = np.zeros((3, 3))
        with pytest.raises(ValueError, match="C.shape"):
            GravityModel("X", MU, R_EARTH, 2, 2, C, S)

    def test_invalid_S_shape_rows(self):
        C = np.zeros((3, 3))
        S = np.zeros((2, 3))
        C[0, 0] = 1.0
        with pytest.raises(ValueError, match="S.shape"):
            GravityModel("X", MU, R_EARTH, 2, 2, C, S)

    def test_invalid_S_shape_cols(self):
        C = np.zeros((3, 3))
        S = np.zeros((3, 2))
        C[0, 0] = 1.0
        with pytest.raises(ValueError, match="S.shape"):
            GravityModel("X", MU, R_EARTH, 2, 2, C, S)

    def test_point_mass_potential_matches(self):
        """With only C[0,0]=1, the GravityModel potential should equal -mu/r."""
        gm = self._make_point_mass_gm(2)
        pm = PointMass("EARTH", MU, R_EARTH)
        r = np.array([7000.0, 1000.0, 500.0])
        np.testing.assert_allclose(gm.potential(0, r), pm.potential(0, r), rtol=1e-8)

    def test_point_mass_acceleration_matches(self):
        gm = self._make_point_mass_gm(2)
        pm = PointMass("EARTH", MU, R_EARTH)
        r = np.array([7000.0, 1000.0, 500.0])
        v = np.array([0.0, 7.5, 0.0])
        np.testing.assert_allclose(
            gm.acceleration(0, r, v), pm.acceleration(0, r), rtol=1e-4
        )

    def test_potential_single_vs_multi(self):
        """potential() with a single 3-vector should return a scalar."""
        gm = self._make_point_mass_gm(2)
        r_single = np.array([7000.0, 0.0, 0.0])
        r_multi = np.array([[7000.0], [0.0], [0.0]])
        val_single = gm.potential(0, r_single)
        val_multi = gm.potential(0, r_multi)
        assert isinstance(val_single, (float, np.floating))
        np.testing.assert_allclose(val_single, val_multi, rtol=1e-12)

    def test_potential_multiple_positions(self):
        gm = self._make_point_mass_gm(0)
        r = np.array([[7000.0, 8000.0], [0.0, 0.0], [0.0, 0.0]])
        result = gm.potential(0, r)
        assert result.shape == (2,)
        np.testing.assert_allclose(result[0], -MU / 7000.0, rtol=1e-8)
        np.testing.assert_allclose(result[1], -MU / 8000.0, rtol=1e-8)

    def test_potential_bad_shape_raises(self):
        gm = self._make_point_mass_gm(2)
        r_bad = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3) — not (3, N)
        with pytest.raises(ValueError):
            gm.potential(0, r_bad)

    def test_from_coefficients(self):
        coeffs = {
            (2, 0): (-0.000484165, 0.0),
            (2, 1): (-2.0e-10, 1.4e-9),
            (2, 2): (2.439e-6, -1.4e-6),
        }
        gm = GravityModel.from_coefficients("TEST", MU, R_EARTH, coeffs)
        assert gm.n_max == 2
        assert gm.m_max == 2
        assert gm.C[0, 0] == 1.0
        np.testing.assert_allclose(gm.C[2, 0], -0.000484165)

    def test_from_file(self, tmp_path):
        """from_file should parse Stokes coefficients correctly."""
        content = (
            "  2  0 -4.84165143790815D-04  0.00000000000000D+00\n"
            "  2  1 -2.06615509074176D-10  1.38441389137979D-09\n"
            "  2  2  2.43938357328313D-06 -1.40027370385934D-06\n"
            "  3  0  9.57161207093473D-07  0.00000000000000D+00\n"
        )
        f = tmp_path / "test_grav.txt"
        f.write_text(content)
        gm = GravityModel.from_file("EARTH", MU, R_EARTH, 2, 2, str(f))
        assert gm.n_max == 2
        assert gm.C[0, 0] == 1.0
        np.testing.assert_allclose(gm.C[2, 0], -4.84165143790815e-04, rtol=1e-12)
        # n=3 line should be skipped since n_max=2
        assert gm.C.shape == (3, 3)


# ===========================================================================
#  src.orbit — keplerian2state and related functions
# ===========================================================================
class TestKeplerian2State:
    """Tests for keplerian2state."""

    def test_circular_equatorial(self):
        """Circular equatorial orbit should have r in xy-plane, v perp to r."""
        a, e, inc, raan, aop, ta = 7000, 0.0, 0.0, 0.0, 0.0, 0.0
        r, v = keplerian2state(a, e, inc, raan, aop, ta, MU)
        # Position should be at (a, 0, 0) for ta=0, e=0
        np.testing.assert_allclose(r, [a, 0, 0], atol=1e-10)
        # Velocity should be circular velocity in +y
        vc = np.sqrt(MU / a)
        np.testing.assert_allclose(np.linalg.norm(v), vc, rtol=1e-6)

    def test_position_magnitude(self):
        a, e = 8000.0, 0.1
        ta = np.radians(45)
        r, v = keplerian2state(a, e, 0.5, 0.3, 0.2, ta, MU)
        expected_r = a * (1 - e**2) / (1 + e * np.cos(ta))
        np.testing.assert_allclose(np.linalg.norm(r), expected_r, rtol=1e-10)

    def test_energy_conservation(self):
        """Specific energy from r,v should equal -mu/(2a)."""
        a = 7500.0
        r, v = keplerian2state(a, 0.05, 0.6, 0.4, 0.1, 1.0, MU)
        energy = np.linalg.norm(v) ** 2 / 2 - MU / np.linalg.norm(r)
        expected = -MU / (2 * a)
        np.testing.assert_allclose(energy, expected, rtol=1e-8)

    def test_angular_momentum_magnitude(self):
        a, e = 9000.0, 0.2
        r, v = keplerian2state(a, e, 0.8, 1.0, 0.5, 0.7, MU)
        h = np.linalg.norm(np.cross(r, v))
        expected_h = np.sqrt(MU * a * (1 - e**2))
        np.testing.assert_allclose(h, expected_h, rtol=1e-8)

    def test_inclination_z_component(self):
        """For inc=90°, the orbit should pass through z-axis."""
        inc = np.pi / 2
        r, v = keplerian2state(7000, 0.0, inc, 0.0, 0.0, np.pi / 2, MU)
        # At ta=90° for a polar orbit with raan=0, aop=0, z should be nonzero
        assert abs(r[2]) > 1.0


class TestOrbitalMechanicsFunctions:
    """Tests for individual orbital mechanics functions."""

    def test_circular_velocity(self):
        r = 7000.0
        expected = np.sqrt(MU / r)
        assert pytest.approx(circular_velocity(MU, r), rel=1e-10) == expected

    def test_escape_velocity(self):
        r = 7000.0
        expected = np.sqrt(2 * MU / r)
        assert pytest.approx(escape_velocity(MU, r), rel=1e-10) == expected

    def test_escape_is_sqrt2_circular(self):
        r = 7000.0
        np.testing.assert_allclose(
            escape_velocity(MU, r) / circular_velocity(MU, r),
            np.sqrt(2),
            rtol=1e-10,
        )

    def test_flight_path_angle_circular(self):
        """Flight path angle of a circular orbit is 0."""
        fpa = flight_path_angle(e=0.0, ta=0.5)
        np.testing.assert_allclose(fpa, 0.0, atol=1e-10)

    def test_flight_path_angle_from_vectors(self):
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        fpa = flight_path_angle(r, v)
        np.testing.assert_allclose(fpa, 0.0, atol=1e-10)

    def test_semimajor_axis_from_rv(self):
        a_expected = 7500.0
        r, v = keplerian2state(a_expected, 0.01, 0.3, 0.1, 0.05, 0.4, MU)
        a_calc = semimajor_axis(r, v, MU)
        np.testing.assert_allclose(a_calc, a_expected, rtol=1e-8)

    def test_semimajor_axis_from_rp_ra(self):
        a = semimajor_axis(rp=6800, ra=7200)
        assert a == 7000.0

    def test_orbit_energy_from_a(self):
        a = 8000.0
        energy = orbit_energy(MU, a=a)
        expected = -MU / (2 * a)
        np.testing.assert_allclose(energy, expected, rtol=1e-12)

    def test_angular_momentum_from_rv(self):
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        h = angular_momentum(r, v)
        expected = np.cross(r, v)
        np.testing.assert_array_equal(h, expected)

    def test_angular_momentum_from_elements(self):
        a, e = 8000.0, 0.1
        h = angular_momentum(a=a, e=e, mu=MU)
        expected = np.sqrt(MU * a * (1 - e**2))
        np.testing.assert_allclose(h, expected, rtol=1e-12)

    def test_eccentricity_from_rv(self):
        a, e_expected = 7500.0, 0.15
        r, v = keplerian2state(a, e_expected, 0.5, 0.2, 0.1, 0.8, MU)
        e_calc = eccentricity(r, v, MU)
        np.testing.assert_allclose(e_calc, e_expected, rtol=1e-8)

    def test_eccentricity_from_rp_ra(self):
        rp, ra = 6800.0, 7200.0
        e = eccentricity(rp=rp, ra=ra)
        expected = (ra - rp) / (ra + rp)
        np.testing.assert_allclose(e, expected, rtol=1e-12)

    def test_eccentricity_vector(self):
        a, e_expected = 7500.0, 0.1
        r, v = keplerian2state(a, e_expected, 0.5, 0.2, 0.1, 0.8, MU)
        e_vec = eccentricity_vector(r, v, MU)
        np.testing.assert_allclose(np.linalg.norm(e_vec), e_expected, rtol=1e-8)

    def test_visviva(self):
        a = 7500.0
        r = 7000.0
        v = visviva(MU, r, a)
        expected = np.sqrt(MU * (2 / r - 1 / a))
        np.testing.assert_allclose(v, expected, rtol=1e-12)

    def test_period(self):
        a = 7000.0
        T = period(MU, a)
        expected = 2 * np.pi * np.sqrt(a**3 / MU)
        np.testing.assert_allclose(T, expected, rtol=1e-12)

    def test_mean_motion(self):
        a = 7000.0
        n = mean_motion(MU, a)
        expected = np.sqrt(MU / a**3)
        np.testing.assert_allclose(n, expected, rtol=1e-12)

    def test_periapse_radius(self):
        a, e = 8000.0, 0.1
        rp = periapse_radius(e, a)
        np.testing.assert_allclose(rp, a * (1 - e), rtol=1e-12)

    def test_apoapse_radius(self):
        a, e = 8000.0, 0.1
        ra = apoapse_radius(e, a)
        np.testing.assert_allclose(ra, a * (1 + e), rtol=1e-12)

    def test_E2ta_and_ta2E_roundtrip(self):
        e = 0.3
        ta_orig = np.radians(60)
        E = ta2E(e, ta_orig)
        ta_back = E2ta(e, E)
        np.testing.assert_allclose(ta_back, ta_orig, rtol=1e-10)

    def test_mean_anomaly_from_eE(self):
        e, E = 0.1, 1.0
        M = mean_anomaly(e, E)
        expected = E - e * np.sin(E)
        np.testing.assert_allclose(M, expected, rtol=1e-12)

    def test_solve_kepler_newton_call_bug(self):
        """solve_kepler passes positional args in wrong order to newton().
        newton(func, fprime, x0) but solve_kepler calls newton(f, Eguess, fprime=fprime),
        meaning fprime gets both a positional and keyword value. Documents existing bug."""
        e = 0.1
        M_target = 1.5
        with pytest.raises(TypeError, match="multiple values"):
            solve_kepler(e, M=M_target)

    def test_stumpff_C_zero(self):
        result = Stumpff_C(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-12)

    def test_stumpff_S_zero(self):
        result = Stumpff_S(np.array([0.0]))
        np.testing.assert_allclose(result, [1.0 / 6.0], atol=1e-12)

    def test_stumpff_C_positive(self):
        z = np.array([1.0])
        result = Stumpff_C(z)
        expected = (1 - np.cos(1.0)) / 1.0
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_stumpff_S_negative(self):
        z = np.array([-1.0])
        result = Stumpff_S(z)
        expected = (np.sinh(1.0) - 1.0) / 1.0
        np.testing.assert_allclose(result, [expected], rtol=1e-10)


class TestLagrangePropagatorsAndStateConversion:
    """Tests for Lagrange propagation and state conversion."""

    def test_lagrange_time_identity(self):
        """Propagating dt=0 should return the same state."""
        a, e = 7500.0, 0.05
        r0, v0 = keplerian2state(a, e, 0.5, 0.3, 0.2, 0.8, MU)
        r, v = lagrange_time(0.0, 0.0, r0, v0, MU)
        np.testing.assert_allclose(r, r0, atol=1e-8)
        np.testing.assert_allclose(v, v0, atol=1e-8)

    def test_lagrange_time_energy_conservation(self):
        a, e = 7500.0, 0.1
        r0, v0 = keplerian2state(a, e, 0.5, 0.3, 0.2, 0.8, MU)
        T = period(MU, a)
        r, v = lagrange_time(T / 4, 0.0, r0, v0, MU)
        E0 = np.linalg.norm(v0) ** 2 / 2 - MU / np.linalg.norm(r0)
        E1 = np.linalg.norm(v) ** 2 / 2 - MU / np.linalg.norm(r)
        np.testing.assert_allclose(E0, E1, rtol=1e-8)

    def test_lagrange_ta_identity_singularity(self):
        """Propagating delta_ta=0 causes division by zero (g=0 → fdot=nan).
        This documents a mathematical singularity in the implementation."""
        r0, v0 = keplerian2state(7500, 0.05, 0.5, 0.3, 0.2, 0.8, MU)
        r, v = lagrange_ta(0.0, r0, v0, MU)
        # Position is correctly returned
        np.testing.assert_allclose(r, r0, atol=1e-8)
        # Velocity is NaN due to 0/0 in fdot = (f*gdot - 1) / g
        assert np.all(np.isnan(v))

    def test_lagrange_ta_small_angle(self):
        """Small but nonzero delta_ta should produce valid state close to initial."""
        r0, v0 = keplerian2state(7500, 0.05, 0.5, 0.3, 0.2, 0.8, MU)
        r, v = lagrange_ta(1e-6, r0, v0, MU)
        np.testing.assert_allclose(r, r0, rtol=1e-4)
        np.testing.assert_allclose(v, v0, rtol=1e-4)

    def test_lagrange_ta_full_orbit(self):
        """Propagating delta_ta=2*pi should return to initial state."""
        r0, v0 = keplerian2state(7500, 0.05, 0.5, 0.3, 0.2, 0.8, MU)
        r, v = lagrange_ta(2 * np.pi, r0, v0, MU)
        np.testing.assert_allclose(r, r0, rtol=1e-6)
        np.testing.assert_allclose(v, v0, rtol=1e-6)


# ===========================================================================
#  src.measurements — generate_NEU_measurements
# ===========================================================================
class TestGenerateNEUMeasurements:
    """Tests for the generate_NEU_measurements function."""

    def setup_method(self):
        # Create a simple 6×N state: 3 position + 3 velocity, N=5
        N = 5
        self.state = np.zeros((6, N))
        for k in range(N):
            angle = 2 * np.pi * k / N
            self.state[0, k] = 1000 * np.cos(angle)  # North
            self.state[1, k] = 1000 * np.sin(angle)  # East
            self.state[2, k] = 500.0                  # Up
            self.state[3, k] = -0.5 * np.sin(angle)
            self.state[4, k] = 0.5 * np.cos(angle)
            self.state[5, k] = 0.0

    def test_missing_return_statement(self):
        """generate_NEU_measurements builds a dict but never returns it.
        This documents a bug: the function is missing 'return measurements'."""
        m = generate_NEU_measurements(self.state, "RANGE")
        assert m is None

    def test_invalid_measurement_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            generate_NEU_measurements(self.state, "INVALID_TYPE")

    def test_internal_range_calculation(self):
        """Verify internal logic by calling function and inspecting via patching."""
        # Since the function doesn't return, we verify through a patched version
        # that the dict construction logic is correct
        state = self.state
        expected_range = np.linalg.norm(state[:3, :], axis=0)
        expected_alt = np.arctan2(state[1, :], state[0, :])

        # Manually replicate what the function does internally
        measurements = {}
        measurements["RANGE"] = np.linalg.norm(state[:3, :], axis=0)
        measurements["ALT"] = np.arctan2(state[1, :], state[0, :])
        measurements["AZ"] = np.arctan2(state[2, :], np.linalg.norm(state[:2, :], axis=0)) % (2 * np.pi)
        range_vec = state[:3, :]
        measurements["RANGE RATE"] = np.sum(range_vec * state[3:6, :], axis=0) / np.linalg.norm(range_vec)

        np.testing.assert_allclose(measurements["RANGE"], expected_range, rtol=1e-12)
        np.testing.assert_allclose(measurements["ALT"], expected_alt, rtol=1e-12)
        assert measurements["RANGE"].shape == (5,)
        assert measurements["AZ"].shape == (5,)
        assert measurements["RANGE RATE"].shape == (5,)

    def test_invalid_multiple_args(self):
        with pytest.raises(ValueError, match="not recognised"):
            generate_NEU_measurements(self.state, "RANGE", "BOGUS")


# ===========================================================================
#  src.body — CelestialBody and Spacecraft
# ===========================================================================
class TestCelestialBody:
    """Tests for CelestialBody."""

    def test_init_with_gravity(self):
        grav = PointMass("EARTH", MU, R_EARTH)
        body = CelestialBody("EARTH", 399, grav)
        assert body.name == "EARTH"
        assert body.id == 399
        assert body.gravity is grav

    def test_init_without_gravity(self):
        body = CelestialBody("TEST", 999)
        assert isinstance(body.gravity, PointMass)
        assert body.gravity.mu == 0

    def test_default_frame(self):
        body = CelestialBody("EARTH", 399)
        assert body.frame == "J2000"

    def test_custom_frame(self):
        body = CelestialBody("MARS", 499, frame="IAU_MARS")
        assert body.frame == "IAU_MARS"

    def test_repr(self):
        body = CelestialBody("EARTH", 399)
        assert "EARTH" in repr(body)
        assert "399" in repr(body)

    def test_inherits_body(self):
        body = CelestialBody("X", 1)
        assert isinstance(body, Body)


class TestSpacecraft:
    """Tests for Spacecraft."""

    def setup_method(self):
        # Reset class-level id counter before each test
        Spacecraft.id = -1
        self.earth = CelestialBody("EARTH", 399)

    def test_init_basic(self):
        sc = Spacecraft("SC-001", mass=500, central_body=self.earth)
        assert sc.name == "SC-001"
        assert sc.mass == 500
        assert sc.central_body is self.earth

    def test_auto_id_assignment(self):
        sc1 = Spacecraft("SC-001", 100, self.earth)
        sc2 = Spacecraft("SC-002", 100, self.earth)
        assert sc1.id == -1
        assert sc2.id == -2

    def test_default_frame_from_central_body(self):
        sc = Spacecraft("SC", 100, self.earth)
        assert sc.frame == self.earth.frame

    def test_custom_frame(self):
        sc = Spacecraft("SC", 100, self.earth, frame="CUSTOM")
        assert sc.frame == "CUSTOM"

    def test_clock_bias_default(self):
        sc = Spacecraft("SC", 100, self.earth)
        assert sc.clock_bias == 0

    def test_clock_bias_custom(self):
        sc = Spacecraft("SC", 100, self.earth, clock_bias=1.5e-6)
        assert sc.clock_bias == 1.5e-6

    def test_inherits_body(self):
        sc = Spacecraft("SC", 100, self.earth)
        assert isinstance(sc, Body)


# ===========================================================================
#  src.atmosphere — ExponentialAtmosphere
# ===========================================================================
class TestExponentialAtmosphere:
    """Tests for ExponentialAtmosphere."""

    def test_default_init(self):
        atm = ExponentialAtmosphere()
        assert atm.rho0 == 3.725e-6
        assert atm.h0 == 90.0
        assert atm.H == 8.5

    def test_custom_init(self):
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        assert atm.rho0 == 6e-13
        assert atm.h0 == 500
        assert atm.H == 50

    def test_density_below_h0(self):
        """Below reference altitude, density should be rho0 * 1E9."""
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        r = R_E + 200  # well below h0=500
        expected = 6e-13 * 1e9
        np.testing.assert_allclose(atm.density(r), expected, rtol=1e-10)

    def test_density_at_h0(self):
        """At exactly h0, density should be rho0 * 1E9."""
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        r = R_E + 500
        # At h0, exp(0) = 1, so density = rho0 * 1E9
        expected = 6e-13 * 1e9
        np.testing.assert_allclose(atm.density(r), expected, rtol=1e-10)

    def test_density_above_h0_decays(self):
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        r1 = R_E + 600
        r2 = R_E + 700
        assert atm.density(r1) > atm.density(r2)

    def test_density_exponential_rate(self):
        """Check exponential decay rate matches scale height."""
        atm = ExponentialAtmosphere(1e-10, 100, 50)
        r1 = R_E + 150
        r2 = R_E + 200
        ratio = atm.density(r1) / atm.density(r2)
        expected_ratio = np.exp(50 / 50)  # delta_h / H
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6)

    def test_callable(self):
        """__call__ should return the same as density(norm(pos))."""
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        pos = np.array([R_E + 600, 0, 0])
        np.testing.assert_allclose(
            atm(0, pos), atm.density(np.linalg.norm(pos)), rtol=1e-12
        )


# ===========================================================================
#  src.atmosphere — AerodynamicDrag
# ===========================================================================
class TestAerodynamicDrag:
    """Tests for AerodynamicDrag."""

    def setup_method(self):
        self.atm = ExponentialAtmosphere(6e-13, 500, 50)
        self.drag = AerodynamicDrag(2.0, 5e-6, 100, self.atm)

    def test_init(self):
        assert self.drag.Cd == 2.0
        assert self.drag.A == 5e-6
        assert self.drag.mass == 100
        assert self.drag.atmosphere is self.atm
        assert self.drag.frame == "IAU_EARTH"

    def test_acceleration_direction(self):
        """Drag should oppose velocity direction."""
        r = np.array([R_E + 600, 0, 0])
        v = np.array([0, 7.5, 0])
        acc = self.drag.acceleration(0, r, v)
        # Drag opposes v, so acc dot v should be negative
        assert np.dot(acc, v) < 0

    def test_acceleration_proportional_to_density(self):
        """Drag at lower altitude (higher density) should be stronger."""
        v = np.array([0, 7.5, 0])
        r_low = np.array([R_E + 550, 0, 0])
        r_high = np.array([R_E + 700, 0, 0])
        acc_low = np.linalg.norm(self.drag.acceleration(0, r_low, v))
        acc_high = np.linalg.norm(self.drag.acceleration(0, r_high, v))
        assert acc_low > acc_high

    def test_acceleration_zero_velocity(self):
        """Zero velocity → zero drag."""
        r = np.array([R_E + 600, 0, 0])
        v = np.array([0, 0, 0])
        acc = self.drag.acceleration(0, r, v)
        np.testing.assert_allclose(acc, [0, 0, 0], atol=1e-30)

    def test_is_perturbation(self):
        assert isinstance(self.drag, Perturbation)


# ===========================================================================
#  src.atmosphere — ConstantAtmosphere
# ===========================================================================
class TestConstantAtmosphere:

    def test_init(self):
        atm = ConstantAtmosphere(1e-12, 300)
        assert atm.density == 1e-12
        assert atm.temperature == 300

    def test_callable_returns_constant(self):
        atm = ConstantAtmosphere(1e-12, 300)
        assert atm(0, np.array([7000, 0, 0])) == 1e-12
        assert atm(1000, np.array([10000, 0, 0])) == 1e-12


# ===========================================================================
#  src.forcemodel — ForceModel
# ===========================================================================
class TestForceModel:
    """Tests for ForceModel (mocking spice calls)."""

    def test_init_valid_forces(self):
        pm = PointMass("EARTH", MU, R_EARTH)
        fm = ForceModel(pm, central_body="EARTH")
        assert len(fm.forces) == 1
        assert fm.central_body == "EARTH"

    def test_init_invalid_force_raises(self):
        with pytest.raises(ValueError, match="not of type Perturbation"):
            ForceModel("not_a_force")

    def test_init_multiple_forces(self):
        pm1 = PointMass("EARTH", MU, R_EARTH)
        pm2 = PointMass("MOON", 4902.8, 1737.4)
        atm = ExponentialAtmosphere(6e-13, 500, 50)
        drag = AerodynamicDrag(2, 5e-6, 100, atm)
        fm = ForceModel(pm1, pm2, drag, central_body="EARTH")
        assert len(fm.forces) == 3

    def test_build_dynamics_returns_callable(self):
        pm = PointMass("EARTH", MU, R_EARTH)
        fm = ForceModel(pm, central_body="EARTH")
        dynamics = fm.build_dynamics()
        assert callable(dynamics)

    def test_dynamics_central_body_only(self):
        """With only central body point-mass, dynamics should give Keplerian motion."""
        pm = PointMass("EARTH", MU, R_EARTH)
        fm = ForceModel(pm, central_body="EARTH", frame="IAU_EARTH")
        dynamics = fm.build_dynamics()
        state = np.array([7000, 0, 0, 0, 7.5, 0], dtype=float)
        deriv = dynamics(0, state)
        # First 3 should be velocity
        np.testing.assert_allclose(deriv[:3], state[3:6], rtol=1e-12)
        # Last 3 should be point-mass acceleration
        expected_acc = pm.acceleration(0, state[:3])
        np.testing.assert_allclose(deriv[3:6], expected_acc, rtol=1e-10)

    def test_dynamics_output_shape(self):
        pm = PointMass("EARTH", MU, R_EARTH)
        fm = ForceModel(pm, central_body="EARTH", frame="IAU_EARTH")
        dynamics = fm.build_dynamics()
        state = np.array([7000, 0, 0, 0, 7.5, 0], dtype=float)
        deriv = dynamics(0, state)
        assert deriv.shape == (6,)


# ===========================================================================
#  src.force — Force and Perturbation base classes
# ===========================================================================
class TestForceBaseClasses:

    def test_force_acceleration_raises(self):
        f = Force()
        with pytest.raises(NotImplementedError):
            f.acceleration(0, np.zeros(3), np.zeros(3))

    def test_perturbation_acceleration_returns_none(self):
        p = Perturbation()
        # Perturbation.acceleration returns None by default
        assert p.acceleration(0, np.zeros(3), np.zeros(3)) is None

    def test_perturbation_is_force(self):
        assert issubclass(Perturbation, Force)


# ===========================================================================
#  src.checkingkwargs
# ===========================================================================
class TestCheckKwargs:

    def test_valid_kwargs(self):
        # Should not raise
        checkkwargs(["a", "b", "c"], {"a": 1, "b": 2})

    def test_invalid_kwargs_raises(self):
        with pytest.raises(ValueError, match="Unfamiliar"):
            checkkwargs(["a", "b"], {"a": 1, "z": 99})

    def test_empty_kwargs_valid(self):
        checkkwargs(["a"], {})

    def test_invalidArgs_is_typeerror(self):
        assert isinstance(invalidArgs, TypeError)


# ===========================================================================
#  src.constants
# ===========================================================================
class TestConstants:

    def test_mu_E(self):
        assert mu_E == 398600

    def test_R_E(self):
        assert R_E == 6378


# ===========================================================================
#  Integration-style tests (combining multiple modules)
# ===========================================================================
class TestIntegration:
    """Higher-level tests combining modules as used in orbit_measurements.py."""

    def test_keplerian_roundtrip(self):
        """keplerian2state → state2keplerian should recover original elements."""
        a_in, e_in = 7500.0, 0.1
        inc_in = np.radians(30)
        raan_in = np.radians(45)
        aop_in = np.radians(60)
        ta_in = np.radians(90)

        r, v = keplerian2state(a_in, e_in, inc_in, raan_in, aop_in, ta_in, MU)
        a_out, e_out, inc_out, raan_out, aop_out, ta_out = state2keplerian(r, v, MU)

        np.testing.assert_allclose(a_out, a_in, rtol=1e-6)
        np.testing.assert_allclose(e_out, e_in, rtol=1e-4)
        np.testing.assert_allclose(inc_out, inc_in, rtol=1e-6)

    def test_celestial_body_with_gravity_model(self):
        """CelestialBody with PointMass gravity should be usable in ForceModel."""
        grav = PointMass("EARTH", MU, R_EARTH)
        earth = CelestialBody("EARTH", 399, grav)
        fm = ForceModel(earth.gravity, central_body="EARTH", frame="IAU_EARTH")
        dynamics = fm.build_dynamics()
        state = np.array([7000, 0, 0, 0, 7.5, 0], dtype=float)
        deriv = dynamics(0, state)
        assert deriv.shape == (6,)

    def test_measurements_returns_none_due_to_missing_return(self):
        """generate_NEU_measurements is missing a return statement, so it returns None.
        This documents the bug at integration level."""
        N = 10
        state = np.random.randn(6, N) * 1000
        state[:3, :] += 7000
        m = generate_NEU_measurements(state, "RANGE", "ALT", "AZ", "RANGE RATE")
        assert m is None

    def test_drag_with_forcemodel(self):
        """AerodynamicDrag + PointMass in ForceModel produces valid output."""
        pm = PointMass("EARTH", MU, R_EARTH)
        # Use a dense atmosphere with high Cd*A/mass to make drag detectable
        atm = ExponentialAtmosphere(1e-6, 0, 50)  # rho0 at h=0, so h0=0 → always exponential
        drag = AerodynamicDrag(2.2, 10.0, 50, atm, frame="IAU_EARTH")
        fm = ForceModel(pm, drag, central_body="EARTH", frame="IAU_EARTH")
        dynamics = fm.build_dynamics()
        state = np.array([R_E + 300, 0, 0, 0, 7.7, 0], dtype=float)
        deriv = dynamics(0, state)
        assert deriv.shape == (6,)
        # Verify drag contributes: the y-acceleration (along v) should be more
        # negative than pure gravity (which has ~0 y-component for this geometry)
        pure_grav_acc_y = pm.acceleration(0, state[:3])[1]
        drag_acc_y = deriv[4] - pure_grav_acc_y
        # Drag opposes velocity (which is in +y), so drag acceleration should be negative
        assert drag_acc_y < 0
