# Tools used to check if kwargs are valid 

invalidArgs = TypeError("The arguments entered are not a valid combination.")

def checkkwargs(acceptable_keywords: list[str] | set[str], kwargs: dict) -> None:
    """Determines if all keyword arguments are valid. If they are not all valid, an error is raised.

    Args:
        acceptable_keywords (list[str] | set[str]): _description_
        kwargs (dict): Dictionary of keyword arguments / value pairs

    Raises:
        TypeError: Unfamiliar arguments provided
    """
    
    bad_kwargs = [keyword for keyword in kwargs.keys() if keyword not in acceptable_keywords]
    if bad_kwargs:
        raise ValueError(f"Unfamiliar arguments provided: {bad_kwargs}")