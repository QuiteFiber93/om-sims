import numpy as np
def euler(func, y0, start, stop, delta_t, *args, **kwargs):
    t = start
    y = y0
    solution = {
        "t" : [start],
        'y' : [y0]
    }
    while t <= stop:
        y += delta_t * func(t, y0, *args, **kwargs)
        t += delta_t
        
        solution['t'].append(t)
        solution['y'].append(y)
        
    solution['t'] = np.array(solution['t'])
    solution['y'] = np.array(solution['y'])
    
    return solution