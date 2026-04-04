from collections import OrderedDict
import numpy as np

class StateDefinition:
    """Defines the layout of a state vector as an ordered collection of named blocks.
    """
    
    def __init__(self, blocks: list[tuple[str, int]] = None):
        self._slices = OrderedDict()
        self._size = 0
        
        if blocks is not None:
            for name, size in blocks:
                self.add(name, size)
                
    @property
    def size(self) -> int:
        """Total number of elements in the state vector."""
        return self._size
    
    @property 
    def blocks(self) -> list[str]:
        """List of block names in insertion order."""
        return list(self._slices.keys())
    
    def add(self, name: str, size: int) -> 'StateDefinition':
        """Add a new block to the state definition. Blocks are appended contiguously in insertion order.
        
        Args:
            name (str): Unique name for this block
            size (int): Number of elements in the block (must be >= 1)
            
        Returns:
            StateDefinition: self, for method chaining
            
        Raises:
            ValueError: If name already exists or size < 1
        """
        if name in self._slices:
            raise ValueError(f"Block '{name}' already exists in state definition.")
        
        if size < 1:
            raise ValueError(f"Block size must be >= 1, got {size}.")
        
        self._slices[name] = slice(self._size, self._size + size)
        self._size += size
        return self
    
    def __getitem__(self, name: str) -> slice:
        """Returns a slice for the named block, for use in array indexing.
        
        Args:
            name (str): Block name
            
        Returns:
            slice: Slice object spanning the block's indices
            
        Raises:
            KeyError: If block name not found
        """
        if name not in self._slices:
            raise KeyError(f"Block '{name}' not found. Available blocks: {self.blocks}")
            
        return self._slices[name]
    
    def has(self, name: str) -> bool:
        """Check whether a named block exists."""
        return name in self._slices
    
    def zeros(self) -> np.ndarray:
        """Returns a zero-initialized state vector of the correct size."""
        return np.zeros(self._size)
    
    def from_dict(self, values: dict) -> np.ndarray:
        """Build a state vector from a dictionary of block names to values.
        
        Args:
            values (dict): Mapping of block names to arrays/scalars.
                           Each value must match the corresponding block size.
                           Blocks not present in the dict are left as zero.
                           
        Returns:
            np.ndarray: Assembled state vector
        """
        state = self.zeros()
        
        for name, val in values.items():
            if name not in self._slices:
                raise KeyError(f"Block '{name}' not found in state definition.")
            
            s = self._slices[name]
            val = np.atleast_1d(val)
            expected = s.stop - s.start
            
            if val.size != expected:
                raise ValueError(
                    f"Block '{name}' expects {expected} elements, got {val.size}."
                )
            
            state[s] = val
            
        return state
    
    def to_dict(self, state: np.ndarray) -> dict:
        """Decompose a state vector into a dictionary of named arrays."""
        return {name: state[s] for name, s in self._slices.items()}
    
    def derivative_map(self, mapping: dict) -> dict:
        """Defines which block's derivative fills which slot in the state derivative.
        
        For dynamics, the time derivative of one block often fills a different block's
        slot. For example, velocity is the derivative of position, so the velocity 
        values fill the position slot of the derivative vector.
        
        Args:
            mapping (dict): Maps block names to their derivative block names.
                            e.g. {"position": "velocity"} means d(position)/dt = velocity
                            
        Returns:
            dict: Maps target block name to (target_slice, source_slice) tuples.
        """
        return {
            target: (self[target], self[source])
            for target, source in mapping.items()
        }
    
    def __contains__(self, name: str) -> bool:
        return name in self._slices
    
    def __len__(self) -> int:
        return len(self._slices)
    
    def __repr__(self):
        lines = [f"StateDefinition (total size: {self._size})"]
        for name, s in self._slices.items():
            lines.append(f"  [{s.start}:{s.stop}] {name} ({s.stop - s.start})")
        return "\n".join(lines)
    
def translational_state() -> StateDefinition:
    """Creates a standard 6-state translational state definition.
    
    Layout: [position(3), velocity(3)]
    """
    return StateDefinition([
        ("position", 3),
        ("velocity", 3),
    ])
 
 
def translational_attitude_state() -> StateDefinition:
    """Creates a 13-state translational + attitude state definition.
    
    Layout: [position(3), velocity(3), quaternion(4), angular_velocity(3)]
    
    The quaternion is scalar-last: [q1, q2, q3, q0] where q0 is the scalar part.
    """
    return StateDefinition([
        ("position", 3),
        ("velocity", 3),
        ("quaternion", 4),
        ("angular_velocity", 3),
    ])
 
 
def estimation_state(include_Cd: bool = False, include_Cr: bool = False) -> StateDefinition:
    """Creates a state definition for estimation (EKF) use.
    
    Starts with translational state and optionally adds estimated parameters.
    """
    sd = translational_state()
    
    if include_Cd:
        sd.add("Cd", 1)
    if include_Cr:
        sd.add("Cr", 1)
        
    return sd