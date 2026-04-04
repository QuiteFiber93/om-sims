"""Tests for StateDefinition class and its integration patterns.

Run with: python -m pytest tests/test_state.py -v
"""
import numpy as np
import pytest
from src.state import (
    StateDefinition,
    translational_state, translational_attitude_state, estimation_state
)


# ---- StateDefinition Core Tests ---- #

class TestStateDefinition:
    
    def test_empty_definition(self):
        sd = StateDefinition()
        assert sd.size == 0
        assert len(sd) == 0
        assert sd.blocks == []
    
    # ---- List constructor tests ---- #
    
    def test_list_constructor(self):
        sd = StateDefinition([
            ("position", 3),
            ("velocity", 3),
        ])
        assert sd.size == 6
        assert sd["position"] == slice(0, 3)
        assert sd["velocity"] == slice(3, 6)
    
    def test_list_constructor_duplicate_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            StateDefinition([("position", 3), ("position", 3)])
    
    def test_list_then_add(self):
        sd = StateDefinition([("position", 3), ("velocity", 3)])
        sd.add("Cd", 1)
        assert sd.size == 7
        assert sd["Cd"] == slice(6, 7)
    
    # ---- add() tests ---- #
        
    def test_add_single_block(self):
        sd = StateDefinition()
        sd.add("position", 3)
        assert sd.size == 3
        assert len(sd) == 1
        assert sd.blocks == ["position"]
        
    def test_add_multiple_blocks(self):
        sd = StateDefinition()
        sd.add("position", 3)
        sd.add("velocity", 3)
        assert sd.size == 6
        assert len(sd) == 2
        
    def test_contiguous_indices(self):
        sd = StateDefinition()
        sd.add("position", 3)
        sd.add("velocity", 3)
        sd.add("quaternion", 4)
        sd.add("angular_velocity", 3)
        
        assert sd["position"] == slice(0, 3)
        assert sd["velocity"] == slice(3, 6)
        assert sd["quaternion"] == slice(6, 10)
        assert sd["angular_velocity"] == slice(10, 13)
        assert sd.size == 13
        
    def test_duplicate_name_raises(self):
        sd = StateDefinition()
        sd.add("position", 3)
        with pytest.raises(ValueError, match="already exists"):
            sd.add("position", 3)
            
    def test_invalid_size_raises(self):
        sd = StateDefinition()
        with pytest.raises(ValueError, match="must be >= 1"):
            sd.add("bad", 0)
            
    def test_missing_key_raises(self):
        sd = StateDefinition()
        sd.add("position", 3)
        with pytest.raises(KeyError, match="not found"):
            sd["nonexistent"]
            
    def test_method_chaining(self):
        sd = StateDefinition()
        result = sd.add("position", 3).add("velocity", 3)
        assert result is sd
        assert sd.size == 6
        
    def test_contains(self):
        sd = StateDefinition()
        sd.add("position", 3)
        assert "position" in sd
        assert "velocity" not in sd
        
    def test_has(self):
        sd = StateDefinition()
        sd.add("position", 3)
        assert sd.has("position") is True
        assert sd.has("velocity") is False


# ---- Array Construction / Decomposition ---- #

class TestStateVectorOperations:
    
    def test_zeros(self):
        sd = translational_state()
        z = sd.zeros()
        assert z.shape == (6,)
        np.testing.assert_array_equal(z, np.zeros(6))
        
    def test_from_dict(self):
        sd = translational_state()
        state = sd.from_dict({
            "position": np.array([7000, 0, 0]),
            "velocity": np.array([0, 7.5, 0]),
        })
        expected = np.array([7000, 0, 0, 0, 7.5, 0])
        np.testing.assert_array_equal(state, expected)
        
    def test_from_dict_partial(self):
        sd = translational_state()
        state = sd.from_dict({"position": np.array([7000, 0, 0])})
        expected = np.array([7000, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(state, expected)
        
    def test_from_dict_wrong_size_raises(self):
        sd = translational_state()
        with pytest.raises(ValueError, match="expects 3 elements"):
            sd.from_dict({"position": np.array([1, 2])})
            
    def test_from_dict_bad_key_raises(self):
        sd = translational_state()
        with pytest.raises(KeyError, match="not found"):
            sd.from_dict({"nonexistent": np.array([1])})
    
    def test_to_dict(self):
        sd = translational_state()
        state = np.array([7000, 0, 0, 0, 7.5, 0])
        d = sd.to_dict(state)
        np.testing.assert_array_equal(d["position"], [7000, 0, 0])
        np.testing.assert_array_equal(d["velocity"], [0, 7.5, 0])
        
    def test_roundtrip(self):
        sd = translational_attitude_state()
        original = {
            "position": np.array([7000, 0, 0]),
            "velocity": np.array([0, 7.5, 0]),
            "quaternion": np.array([0, 0, 0, 1]),
            "angular_velocity": np.array([0.01, 0, 0]),
        }
        state = sd.from_dict(original)
        recovered = sd.to_dict(state)
        for key in original:
            np.testing.assert_array_equal(recovered[key], original[key])

    def test_scalar_block_from_dict(self):
        sd = estimation_state(include_Cd=True)
        state = sd.from_dict({
            "position": np.array([7000, 0, 0]),
            "velocity": np.array([0, 7.5, 0]),
            "Cd": np.array([2.2]),
        })
        assert state[sd["Cd"]][0] == 2.2
        assert sd.size == 7


# ---- Indexing Into Real State Vectors ---- #

class TestStateIndexing:
    
    def test_translational_indexing(self):
        sd = translational_state()
        state = np.array([7000.0, 100.0, 50.0, -0.5, 7.2, 0.3])
        
        r = state[sd["position"]]
        v = state[sd["velocity"]]
        
        np.testing.assert_array_equal(r, [7000, 100, 50])
        np.testing.assert_array_equal(v, [-0.5, 7.2, 0.3])
        
    def test_attitude_indexing(self):
        sd = translational_attitude_state()
        state = np.zeros(13)
        state[sd["quaternion"]] = [0, 0, 0, 1]
        state[sd["angular_velocity"]] = [0.01, 0, 0]
        
        np.testing.assert_array_equal(state[sd["quaternion"]], [0, 0, 0, 1])
        np.testing.assert_array_equal(state[sd["angular_velocity"]], [0.01, 0, 0])
        
    def test_augmented_state_indexing(self):
        sd = estimation_state(include_Cd=True, include_Cr=True)
        state = np.zeros(sd.size)
        state[sd["Cd"]] = 2.2
        state[sd["Cr"]] = 1.5
        
        assert state[6] == 2.2
        assert state[7] == 1.5
        assert sd.size == 8
        
    def test_write_through_slice(self):
        sd = translational_state()
        state = np.zeros(6)
        state[sd["velocity"]] = [0, 7.5, 0]
        np.testing.assert_array_equal(state, [0, 0, 0, 0, 7.5, 0])


# ---- Factory Function Tests ---- #

class TestFactories:
    
    def test_translational_state(self):
        sd = translational_state()
        assert sd.size == 6
        assert sd.blocks == ["position", "velocity"]
        
    def test_translational_attitude_state(self):
        sd = translational_attitude_state()
        assert sd.size == 13
        assert sd.blocks == ["position", "velocity", "quaternion", "angular_velocity"]
        
    def test_estimation_state_base(self):
        sd = estimation_state()
        assert sd.size == 6
        
    def test_estimation_state_with_Cd(self):
        sd = estimation_state(include_Cd=True)
        assert sd.size == 7
        assert sd.has("Cd")
        
    def test_estimation_state_with_both(self):
        sd = estimation_state(include_Cd=True, include_Cr=True)
        assert sd.size == 8
        assert sd["Cd"] == slice(6, 7)
        assert sd["Cr"] == slice(7, 8)


# ---- Derivative Map Tests ---- #

class TestDerivativeMap:
    
    def test_basic_derivative_map(self):
        sd = translational_state()
        dmap = sd.derivative_map({"position": "velocity"})
        
        target_slice, source_slice = dmap["position"]
        assert target_slice == slice(0, 3)
        assert source_slice == slice(3, 6)
        
    def test_derivative_map_usage(self):
        sd = translational_state()
        state = np.array([7000, 0, 0, 0, 7.5, 0], dtype=float)
        dstate = np.zeros(sd.size)
        
        dmap = sd.derivative_map({"position": "velocity"})
        target_slice, source_slice = dmap["position"]
        dstate[target_slice] = state[source_slice]
        
        np.testing.assert_array_equal(dstate[:3], [0, 7.5, 0])


# ---- Repr Tests ---- #

class TestRepr:
    
    def test_repr_format(self):
        sd = translational_state()
        r = repr(sd)
        assert "total size: 6" in r
        assert "position" in r
        assert "velocity" in r
        assert "[0:3]" in r
        assert "[3:6]" in r