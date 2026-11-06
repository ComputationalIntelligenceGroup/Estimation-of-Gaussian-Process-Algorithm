### Non-linear forced harmonic oscillator problem (Non-linear SE):
##    Non linear ODE y(x):
##       
##     1/2 y'' - 1/2 y + y^3 = 0
##
##    Boundary conditions: 
##     y(0) = 1;  y'(0) = 0
##
##    Conditions on y(x): are imposed as evidences on the Gaussian Process itself
##    Conditions on y'(x) are imposed inside cost function with term e^|y'candidate_sol(0) - y'(0)| - 1
##
## We are looking for a discretized version of solution y(x) = [y(x1), y(x2), ..., y(xn)]    
## Numerical second derivative reads:
## y''(x{i}) = (y(x{i-1}) + y (x{i+1}) - 2 y(x{i})) / h^2  with h = x{i}-x{i-1} = cte in our case with fixed grid
##
## Numerical first derivative (for constraint on y'(x)):
## y'(x{i}) = (y(x{i+1}) - y(x{i-1})) / 2h

import numpy as np
from numpy.linalg import norm
from typing import Union

def NLSE(x:Union[np.array, list],  L:float = 2, exponential:bool = False):

    n = len(x)
    beta = 1
    R = np.float128(0)
    h = L/(n-1) #spacing of x coordinate
    ## Residuals of instance x on equation = 0
    for i in range (n):
            if i==0:
                value = abs(0.5*((x[i]-2*x[i+1]+x[i+2])/h**2) - 0.5*x[i] + x[i]**3)
                assert value >= 0
                R += value

            elif i==n-1:
                value = abs(0.5*((x[i]+x[i-2]-2*x[i-1])/h**2) - 0.5*x[i] + x[i]**3)
                assert value >= 0
                R += value
               
            else:
                value = abs(0.5*((x[i+1]+x[i-1]-2*x[i])/h**2) - 0.5*x[i] + x[i]**3)
                assert value >= 0
                R += value

    ## Derivative conditions y'(0) = 0. If number of grid points n is odd, and the problem is defined on the interval [-1, 1]. The index i identifying y(0) is:
    ##   floor(n/2) = n//2 because indexes start from 0. Applying numerical derivative again on that point y'(0):
    ## first_derivative fd:
    fd = np.float128((x[n//2+1]-x[n//2-1])/(2*h))
    D = np.exp(abs(fd - 0)) - 1
    
    ## Boundary condition as penalization term
    pen = 0
    if (x[n//2] < 1 - 1e-4 or x[n//2] > 1 + 1e-4):
        #pen = 10e3
        pen = np.exp(abs(x[n//2] - 1)) - 1


    if exponential is False:
        Z = 100*R + 10000*D + pen#cost function always positive, best solution where Z = 0
    #print(Z, '\n', R, '\n', D, '\n\n\n')
    if exponential is True:
        Z = 100*R + D + pen #cost function always positive, best solution where Z = 0
        Z = -np.exp(-10**-2*Z) #exponential of cost function defined [0, -1] optimum in -1
    return Z

# for i in range(n):
    #         if i==0:
    #             R = R -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
    #             j = j + x[i]**2* (L/(n-1))

    #         elif i==n-1:
    #             R = R -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
    #             j = j + x[i]**2 * (L/(n-1))
                
    #         else:
    #             R = R -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
    #             j=j+x[i]**2* (L/(n-1))