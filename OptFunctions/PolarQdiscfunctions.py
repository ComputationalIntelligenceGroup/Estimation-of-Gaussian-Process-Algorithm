import numpy as np
from numpy.linalg import norm
from typing import Union

def Qdisc(Psi:Union[np.array, list], radio:Union[np.array, list], dr = float, dtheta = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(r_i, theta_j). Rows represent amplitudes for fixed ri and varying (columns j) discretized angles.
    # Columns represent amplitudes for fixed angle values theta j and varying rows (i) radius values.
    
    #Radio is an array representing the discretized values of the r variable. radio = [r0, r1, ..., rn]

    #Derivatives of Psi calculation
    dfdr = np.gradient(Psi, dr, axis=0)
    dfdt = np.gradient(Psi, dtheta,axis = 1)
    frr = np.gradient(dfdr, dr, axis=0)
    ftt = np.gradient(dfdt, dtheta, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian in polar coordinates which is
    #Laplacian = S1 + S2 + S3 a sum of three terms

    for row in range(len(dfdr)):
        S1 = dfdr[row, :]/radio[row]
        S3 = ftt[row, :]/ (radio[row]**2)

    L = S1 + frr + S3

    #Integrand I is Psi*L multiplied by Jacobian determinant rdrdth which takes into account r value for every point in function.
    #Then we sum all the points in the integrand
    #El integrando I es Psi*L y luego habrá que tener en cuenta el determinante jacobiano para cada punto y sumar todos los puntos
    I = Psi*L

    #Now the Jacobian determinant
    #Determinante Jacobiano que es rdrdth con lo que cada punto de la función discretizada se multiplica por r en cada punto
    for row in range(len(I)):
        I[row, :] = I[row, :]*radio[row]

    #Una vez multiplicado cada punto de la función discretizada por su jacobiano solo queda sumar todos los elementos multiplicados por drdth que es cte 
    #Finally we only sum all the elements in the integrand multiplied by drdth (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I*dr*dtheta)
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psi2 = Psi * Psi
    for row in range(len(Psi)):
        Psi2[row, :] = Psi2[row, :]*radio[row]

    norm = np.sum(Psi2*dr*dtheta) #If states are normalized this equals 1

    #Finally the optimization function based on the expected energy
    Z = (ExpE/norm - E)**2

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


def Qdisc1(Psi:Union[np.array, list], Psi0:Union[np.array, list], E0:float, radio:Union[np.array, list], dr = float, dtheta = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = E0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(r_i, theta_j). Rows represent amplitudes for fixed ri and varying (columns j) discretized angles.
    # Columns represent amplitudes for fixed angle values theta j and varying rows (i) radius values.
    
    #Radio is an array representing the discretized values of the r variable. radio = [r0, r1, ..., rn]

    #Derivatives of Psi calculation
    dfdr = np.gradient(Psi, dr, axis=0)
    dfdt = np.gradient(Psi, dtheta,axis = 1)
    frr = np.gradient(dfdr, dr, axis=0)
    ftt = np.gradient(dfdt, dtheta, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian in polar coordinates which is
    #Laplacian = S1 + S2 + S3 a sum of three terms

    for row in range(len(dfdr)):
        S1 = dfdr[row, :]/radio[row]
        S3 = ftt[row, :]/ (radio[row]**2)

    L = S1 + frr + S3

    #Integrand I is Psi*L multiplied by Jacobian determinant rdrdth which takes into account r value for every point in function.
    #Then we sum all the points in the integrand
    #El integrando I es Psi*L y luego habrá que tener en cuenta el determinante jacobiano para cada punto y sumar todos los puntos
    I = Psi*L

    #Now the Jacobian determinant
    #Determinante Jacobiano que es rdrdth con lo que cada punto de la función discretizada se multiplica por r en cada punto
    for row in range(len(I)):
        I[row, :] = I[row, :]*radio[row]

    #Una vez multiplicado cada punto de la función discretizada por su jacobiano solo queda sumar todos los elementos multiplicados por drdth que es cte 
    #Finally we only sum all the elements in the integrand multiplied by drdth (constant for every point) and -1/2
    # which appears in the Schroedinger equation

    ExpE = -0.5*np.sum(I*dr*dtheta)
    
    #Another term in the optimization function is the norm of the state
    #Norm of state
    Psi2 = Psi * Psi
    for row in range(len(Psi)):
        Psi2[row, :] = Psi2[row, :]*radio[row]

    norm = np.sum(Psi2*dr*dtheta) #If state is normalized this equals 1

    #Orthogonal term for excitated states
    Psi2 = Psi * Psi0
    for row in range(len(Psi)):
        Psi2[row, :] = Psi2[row, :]*radio[row]

    orto = np.sum(Psi2*dr*dtheta) #If states are orthogonal this equals 0

    #Finally the optimization function based on the expected energy
    Z = (ExpE/norm - E)**2 + 100*orto**2

    if exponential is True:
        Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    return Z



def QdiscSimple(Psi:Union[np.array, list], radio:Union[np.array, list], dr = float, dtheta = float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Psi is a 2D array (matrix) so that Psi_ij = Psi(r_i, theta_j). Rows represent amplitudes for fixed ri and varying (columns j) discretized angles.
    # Columns represent amplitudes for fixed angle values theta j and varying rows (i) radius values.
    
    #Radio is an array representing the discretized values of the r variable. radio = [r0, r1, ..., rn]

    #Derivatives of Psi calculation
    dfdr = np.gradient(Psi, dr, axis=0)
    dfdt = np.gradient(Psi, dtheta,axis = 1)
    frr = np.gradient(dfdr, dr, axis=0)
    ftt = np.gradient(dfdt, dtheta, axis=1)

    #Expected energy calculation:

    #First we calculate Laplacian in polar coordinates which is
    #Laplacian = S1 + S2 + S3 a sum of three terms

    for row in range(len(dfdr)):
        S1 = dfdr[row, :]/radio[row]
        S3 = ftt[row, :]/ (radio[row]**2)

    L = S1 + frr + S3

    #Optimization function based on sum of residues 
    Z = np.abs(L + 2*E*Psi)
    Z = np.sum(Z)

    # #Finally the optimization function based on the expected energy
    # Z = (ExpE/norm - E)**2

    # if exponential is True:
    #     Z = -np.exp(-10*Z) #Función acotada entre -1 y 0. Óptimo en -1
        
    # # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z
