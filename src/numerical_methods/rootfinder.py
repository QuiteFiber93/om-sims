import numpy as np

def newton(func, fprime, x0, tol = 1E-10, max_iter = 1000):
    x = x0
    oldx = x0
    n = 1
    err = np.inf
    while abs(err) > tol and n <= max_iter:
        oldx = x
        x = x - func(x) / fprime(x)
        err = abs(x - oldx)
        n += 1
        
    if abs(err) > tol and n > max_iter:
        raise RuntimeError('Solution did not conver before reaching the maximum iteration count.')
    
    return x
