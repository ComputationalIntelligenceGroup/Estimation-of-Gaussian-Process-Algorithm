import numpy as np
from numpy.linalg import norm
from typing import Union

def TwoDBox(Psi:Union[np.array, list], dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.

    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    #Finally the optimization function based on the expected energy
    Z = (ExpE/norm - E)**2

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z



def TwoDBoxExc1(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.

    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    
    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    Psiorto1 = Psi * Psi0.T
    orto1 = np.sum(Psiorto1*dx*dy)

    #Finally the optimization function based on the expected energy
    Z = (ExpE/norm - E)**2 + 1000*orto**2 + 1000*orto1**2


    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def TwoDBoxExc12(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, Psi1:Union[np.array, list], E1:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E0 #+ E1
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.


    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto1 = Psi * Psi1
    orto1 = np.sum(Psiorto1*dx*dy) #If states are orthogonal this equals 0

    #Finally the optimization function based on the expected energy
    Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2


    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def TwoDBoxExc2(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, Psi1:Union[np.array, list], E1:float, Psi12:Union[np.array, list], E12:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound (the energy of the previous state)
    E = E1
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.


    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto1 = Psi * Psi1
    orto1 = np.sum(Psiorto1*dx*dy) #If states are orthogonal this equals 0

    Psiorto12 = Psi * Psi12
    orto12 = np.sum(Psiorto12*dx*dy) #If states are orthogonal this equals 0

    Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto12**2

    ##############################################################################
    ## The following code was used in the case where Psi12 was optionally given ##
    ##############################################################################
    #Orthogonal term for transpose of Psi 1
    # Psiorto2 = Psi * Psi1.T #Psi1.T corresponds to the opposite (1,2) or (2,1) which are orthogonal states with the same energy
    # orto2 = np.sum(Psiorto2*dx*dy) #If states are orthogonal this equals 0

    # #If Psi12 is given (the orthogonal of Psi1 with the same energy) additional orthogonal terms are added
    # if Psi12 is not None:
    #     #Orthogonal term for excitated states
    #     Psiorto12 = Psi * Psi12
    #     orto12 = np.sum(Psiorto12*dx*dy) #If states are orthogonal this equals 0

    #     #Orthogonal term for excitated states
    #     Psiorto12T = Psi * Psi12.T #Psi1.T corresponds to the opposite (1,2) or (2,1) which are orthogonal states with the same energy
    #     orto12T = np.sum(Psiorto12T*dx*dy) #If states are orthogonal this equals 0
            
    #     #Finally the optimization function based on the expected energy
    #     Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto2**2 + 10000*orto12**2 + 10000*orto12T**2

    # else:
    #     #Finally the optimization function based on the expected energy
    #     Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto2**2
    #####################################################################################
    #####################################################################################

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def TwoDBoxExc3(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, Psi1:Union[np.array, list], E1:float, Psi12:Union[np.array, list], E12:float, Psi2:Union[np.array, list], E2:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E2
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.


    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto1 = Psi * Psi1
    orto1 = np.sum(Psiorto1*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for transpose of Psi 1
    Psiorto12 = Psi * Psi12
    orto12 = np.sum(Psiorto12*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto22 = Psi * Psi2
    orto22 = np.sum(Psiorto22*dx*dy) #If states are orthogonal this equals 0

    Z = (ExpE/norm - E)**2 + 1000000*orto22**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto12**2
    
    ##############################################################################
    ## The following code was used in the case where Psi12 was optionally given ##
    ##############################################################################
    #Orthogonal term for transpose of Psi 1
    # Psiorto22T = Psi * Psi2.T 
    # orto22T = np.sum(Psiorto22T*dx*dy) #If states are orthogonal this equals 0

    # #If Psi12 is given (the orthogonal of Psi1 with the same energy) additional orthogonal terms are added
    # if Psi12 is not None:
    #     #Orthogonal term for excitated states
    #     Psiorto12 = Psi * Psi12
    #     orto12 = np.sum(Psiorto12*dx*dy) #If states are orthogonal this equals 0

    #     #Orthogonal term for excitated states
    #     Psiorto12T = Psi * Psi12.T #Psi1.T corresponds to the opposite (1,2) or (2,1) which are orthogonal states with the same energy
    #     orto12T = np.sum(Psiorto12T*dx*dy) #If states are orthogonal this equals 0
            
    #     #Finally the optimization function based on the expected energy
    #     Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto2**2 + 10000*orto22**2 + 10000*orto22T + 10000*orto12**2 + 10000*orto12T**2

    # else:
    #     #Finally the optimization function based on the expected energy
    #     Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto2**2 + 10000*orto22**2 + 10000*orto22T
    ##########################################################################################
    ##########################################################################################

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def TwoDBoxExc31(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, Psi1:Union[np.array, list], E1:float, Psi12:Union[np.array, list],  E12:float, Psi2:Union[np.array, list], E2:float, Psi3:Union[np.array, list], E3:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E2
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.


    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    #Integrand I is Psi*L
    #Then we will sum all the points in the integrand
    I = Psi*L

    #Finally we only sum all the elements in the integrand multiplied by dxdy (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I)*dx*dy
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psipsi = Psi * Psi

    norm = np.sum(Psipsi*dx*dy) #If states are normalized this equals 1

    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto1 = Psi * Psi1
    orto1 = np.sum(Psiorto1*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for transpose of Psi 1
    Psiorto12 = Psi * Psi12
    orto12 = np.sum(Psiorto12*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto22 = Psi * Psi2
    orto22 = np.sum(Psiorto22*dx*dy) #If states are orthogonal this equals 0

    #Orthogonal term for excitated states
    Psiorto3 = Psi * Psi3
    orto3 = np.sum(Psiorto3*dx*dy) #If states are orthogonal this equals 0

    Z = (ExpE/norm - E)**2 + 10000*orto**2 + 10000*orto1**2 + 10000*orto12**2 + 10000*orto22**2 + 10000*orto3**2
    

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def TwoDBoxSimple(Psi:Union[np.array, list], dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.

    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #First we calculate Laplacian
    L = fxx + fyy

    r = abs(-0.5*L-E*Psi)
    R = np.sum(r)
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return R


def TwoDBoxExc1Simple(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, dx = float, dy = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(x_i, y_j). Rows represent amplitudes for fixed xi and varying (columns j) yj.
    # Columns represent amplitudes for fixed y values and varying rows (i) x values.

    #Derivatives of Psi calculation
    dfdx = np.gradient(Psi, dx, axis=0)
    dfdy = np.gradient(Psi, dy,axis = 1)
    fxx = np.gradient(dfdx, dx, axis=0)
    fyy = np.gradient(dfdy, dy, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian
    L = fxx + fyy

    r = abs(-0.5*L-E*Psi)
    R = np.sum(r)

    #Orthogonal term for excitated states
    Psiorto = Psi * Psi0
    orto = np.sum(Psiorto*dx*dy) #If states are orthogonal this equals 0

    Psiorto1 = Psi * Psi0.T
    orto1 = np.sum(Psiorto1*dx*dy)

    #Finally the optimization function based on the expected energy
    Z = R + 1000*orto**2 + 1000*orto1**2
        
    return Z

