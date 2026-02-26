from numpy import pi
from src.checkingkwargs import invalidArgs, checkkwargs

def cannonical_units(quantity: float, nondim_unit: str, **kwargs) -> float:
    """Redefines quantities in terms of canconical units. Currently implemented units are TU, DU, and MU

    Args:
        quantity (float): Quantity to be nondimensionalized
        nondim_unit (str): Desired nondimensional units.

    Raises:
        invalidArgs: Invalid keyword arguments or nondimensional quantity

    Returns:
        float: _description_
    """
    # checking if kwargs are valid
    acceptable_keywords = ['period', 'a', 'm1', 'm2']
    checkkwargs(acceptable_keywords, kwargs)
    
    # removing whitespace from nondim_unit and converts to uppercase
    # needed so the dimensions are checked properly in the if/elif statements below
    nondim_unit = "".join(nondim_unit.split()).upper()
    
    # accessing keys so the function does not have to be called frequently
    keys = kwargs.keys()
    
    # Checking conversion
    if set(['period']) <= keys and nondim_unit == "TU":
        period = kwargs['period']
        return quantity*2*pi/period
    
    elif set(['a']) <= keys and nondim_unit == 'DU':
        a = kwargs['a']
        return quantity/a
    
    elif set(['m1', 'm2']) <= keys and nondim_unit == 'MU':
        m1 = kwargs['m1']
        m2 = kwargs['m2']
        
        MU_conversion = 1/(m1 + m2)
        return quantity*MU_conversion
    
    elif set(['a', 'period']) <= keys and nondim_unit in ['DU/TU', 'VU']:
        a = kwargs['a']
        period = kwargs['period']
        
        TU_conversion = 2*pi/period
        DU_conversion = 1/a
        
        return quantity*DU_conversion/TU_conversion

    elif set(['a', 'period']) <= keys and nondim_unit == 'DU^3/TU^2':
        a = kwargs['a']
        period = kwargs['period']
        
        TU_conversion = 2*pi/period
        DU_conversion = 1/a
        
        return quantity * DU_conversion**3 / TU_conversion**2
    
    else:
        raise invalidArgs