import numpy as np
from numpy.linalg import norm
from typing import Union

def Qdisc(Psi:Union[np.array, list],  mask: np.array, grid_size = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Disc radius, is implicitly defined in mask
    #mask is a numpy array indicating the elements of Psi inside the disc
    #N = 101  #Discretization size per variable
    # x = np.linspace(-1, 1, N)
    # y = np.linspace(-1, 1, N)
    # X, Y = np.meshgrid(x, y)
    # mask to identify points inside the disc
    # mask = X**2 + Y**2 <= 1
    
    #First we calculate the first derivatives of the input discretized function
    Psi_x, Psi_y = np.gradient(Psi, grid_size)
    #Second derivatives
    Psi_xx = np.gradient(Psi_x, grid_size, axis=0)
    Psi_yy = np.gradient(Psi_y, grid_size, axis=1)
    #Now we truncate the points outside the circle (which are known to be zero everywhere)
    Psi = Psi[mask]
    Psi_xx = Psi_xx[mask]
    Psi_yy = Psi_yy[mask]

    #Now the integral ExpE
    Integrand = -0.5 * Psi*[Psi_xx + Psi_yy]
    Integral = grid_size**2*np.sum(Integrand)

    #Norm of state
    norm = grid_size**2*np.sum(Psi*Psi)

    #Finally the optimization function based on the expected energy
    Z = (Integral/norm - E)**2
    if exponential is True:
        Z = -np.exp(-Z) #Función acotada entre -1 y 0. Óptimo en -1
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z

