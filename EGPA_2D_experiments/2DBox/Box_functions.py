import numpy as np
from typing import Union

def reconvert_points_from_cartesian_stack_to_2D(SAMPLES, x, y):
    # Reconversión de puntos proceso gaussiano a 2d array que representa valor de la función Psi_ij = Psi(x_i, y_j)
    # Toma un array de 2D (puntos en cartesianas, individuos)
    # y lo convierte en una array 3D (individuo, matriz 2D (xi, yj)) que representa una generación
    solution = None
    for i in range(SAMPLES.shape[1]):
        ind = SAMPLES[:, i].reshape(len(x), len(y))
        #ind = np.expand_dims(ind, axis=0)
        if solution is None:
            solution = ind
        else:
            solution = np.concatenate([solution, ind])
    return solution

def EnergyBox(Psi:Union[np.array, list], dx = float, dy = float) -> float:
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
    return ExpE


def State_Norm(Psi, dx, dy):
    #Función que devuelve integral de norma de un estado. Estado Psi es una matriz 2d Psi_ij = Psi(r_i, th_j)
    Psipsi = Psi * Psi
    integral = np.sum(Psipsi*dx*dy)
    #Psi = Psi/np.sqrt(integral)
    return integral


def Normalize(Psi, dx, dy):
    #Función que normaliza una solución para que su cuadrado integre a 1. La solución es una matriz 2d Psi_ij = Psi(r_i, th_j)
    Psipsi = Psi * Psi
    integral = np.sum(Psipsi*dx*dy)
    Psi = Psi/np.sqrt(integral)
    return Psi

def Psinxny(nx, ny, x, y, dx, dy):
    L=1
    Funcion = None
    for i in x:
        for j in y:
            valor = 2/L*np.sin(nx*np.pi*i/L)*np.sin(ny*np.pi*j/L)
            if Funcion is None:
                Funcion = valor
            else:
                Funcion = np.append(Funcion, valor)
    Estado = np.reshape(Funcion, (len(Funcion), 1))
    Estado = reconvert_points_from_cartesian_stack_to_2D(SAMPLES=Estado, x=x, y=y)
    Ener = EnergyBox(Estado, dx=dx, dy=dy)
    return Funcion.flatten(), Estado, Ener

def Enxny(nx, ny):
    L=1
    return np.pi**2/(2*L**2)*(nx**2+ny**2)   