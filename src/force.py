class Force:
    """
    Base class for forces and perturbations. 
    """
    def __init__(self):
        pass
    
    def acceleration(self, t, r, v):
        raise NotImplementedError("Subclasses of Perturbation must implement acceleration()")
    
class Perturbation(Force):
    """
    Base class for perturbations to dynamics. 
    """
    def acceleration(self, t, r, v):
        pass