import numpy as np

def generate_NEU_measurements(state: np.ndarray, *args):
    valid_measurements = set(["RANGE", "ALT", "AZ", "RANGE RATE"])
    bad_args = [arg for arg in args if arg.upper() not in valid_measurements]
    if bad_args:
        raise ValueError(f"Arguments not recognised for this funtion: {bad_args}. Acceptable arguments include {valid_measurements}")
    
    # setting up dict for measurement returns
    measurements = {}
    for arg in args:
        if arg.casefold() == "RANGE".casefold():
            measurements[arg] = np.linalg.norm(state[:3, :], axis = 0)
        
        elif arg.casefold() == "ALT".casefold():
            measurements[arg] = np.arctan2(state[1, :], state[0, :])
        
        elif arg.casefold() == "AZ".casefold():
            measurements[arg] = np.arctan2(state[2, :], np.linalg.norm(state[:2, :], axis = 0)) % (2 * np.pi)
            
        elif arg.casefold() == "RANGE RATE".casefold():
            range_vec = state[:3, :]
            measurements[arg] = np.sum(range_vec * state[3:6, :], axis = 0) / np.linalg.norm(range_vec)
        
        else:
            raise ValueError("Measurement model not set up for " + arg)