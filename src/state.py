class StateBlock:
    def __init__(self, name: str, start: int, size: int):
        self.name = name
        self.start = start
        self.size = size
        self.stop = start + size
    
    @property
    def indices(self) -> slice:
        """Returns a slice object for indexing into the state vector."""
        return slice(self.start, self.stop)
    
    def __repr__(self):
        return f"StateBlock('{self.name}', size={self.size}, indices=[{self.start}:{self.stop}])"