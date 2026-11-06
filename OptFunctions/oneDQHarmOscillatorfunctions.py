import numpy as np
from numpy.linalg import norm
from typing import Union


#Definición de la función de optimización basada en referencia: Direct solution of Schrödinger equation by genetic algorithm:
#test cases
#########FUNDAMENTAL STATE######################

#Definición de la función de optimización basada en referencia: Direct solution of Schrödinger equation by genetic algorithm:
#test cases
###############################
def oneDHOExpE(y:Union[np.array, list], x:Union[np.array, list], exponential:bool = False) -> float:
    "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
    #Energy_lower bound
    E = 0
    #grid size x=[-4, 4]
    n = len(y)
    #x=np.linspace(-4, 4, n)
    dx = (x[-1]-x[0])/(n-1)
    #integral sums
    ExpE = 0; j=0
    cons = 10
    # if (y[0]< -0.1 or y[0] > 0.1) or (y[n-1]<-0.1 or y[n-1]>0.1):
    #     F = np.inf
    #     return F
    
    ## Boundary condition as penalization term
    pen = 0
    if (y[0] < -1e-4 or y[0] >1e-4) or (y[-1] < -1e-4 or y[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(y[0] - 0)) - 1
        D1 = np.exp(abs(y[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i+2]-2*y[i+1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
        
        elif i==n-1:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i-2]-2*y[i-1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
        
        else:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i-1]+y[i+1]-2*y[i])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
    if j > 10e1:
        j=10e1
    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + pen
    if exponential is True:
        Z = -np.exp(-cons*Z) #Función acotada entre -1 y 0. Óptimo en -1
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z



########## 1er  estado excitado  #####################
def oneDHOExpE1(y:Union[np.array, list], x:Union[np.array, list], E0:float, Psi0:Union[np.array, list] ,exponential:bool = False) -> float:
    "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
    #Energy_lower bound
    E = E0 #siempre es la suma de las anteriores pero éstas dependen de la definición última de la función de coste (exponencial o no). results_dict['eda_resultExpE'].sel_inds_hist[-1, 0, -1]**0.5 #valor esperado energía n=0
    #grid size x=[-4, 4]
    n = len(y)
    #x=np.linspace(-4, 4, n)
    dx = (x[-1]-x[0])/(n-1)
    #integral sums
    ExpE = 0; j=0; orto=0
    cons=10
    # if (y[0]< -0.1 or y[0] > 0.1) or (y[n-1]<-0.1 or y[n-1]>0.1):
    #     F = np.inf
    #     return F

    ## Boundary condition as penalization term
    pen = 0
    if (y[0] < -1e-4 or y[0] >1e-4) or (y[-1] < -1e-4 or y[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(y[0] - 0)) - 1
        D1 = np.exp(abs(y[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i+2]-2*y[i+1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
        
        elif i==n-1:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i-2]-2*y[i-1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
        
        else:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i-1]+y[i+1]-2*y[i])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
    if j > 10e1:
        j=10e1
    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + 10*orto**2 + pen
    if exponential is True:
        Z = -np.exp(-cons*Z) #Función acotada entre -1 y 0. Óptimo en -1
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z

########## 2o  estado excitado  #####################
def oneDHOExpE2(y:Union[np.array, list], x:Union[np.array, list], E0:float, Psi0:Union[np.array, list],  E1:float, Psi1:Union[np.array, list], exponential:bool = False) -> float:
    "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
    #Energy_lower bound
    E = E0+E1 #results_dict['eda_resultExpE1'].sel_inds_hist[-1, 0, -1]**0.5 + results_dict['eda_resultExpE'].sel_inds_hist[-1,0,-1]**0.5 #valor esperado energía n=1
    #grid size x=[-4, 4]
    n = len(y)
    #x=np.linspace(-4, 4, n)
    dx = (x[-1]-x[0])/(n-1)
    #integral sums
    ExpE = 0; j=0; orto=0; orto1=0
    cons=10
    # if (y[0]< -0.1 or y[0] > 0.1) or (y[n-1]<-0.1 or y[n-1]>0.1):
    #     F = np.inf
    #     return F
    ## Boundary condition as penalization term
    pen = 0
    if (y[0] < -1e-4 or y[0] >1e-4) or (y[-1] < -1e-4 or y[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(y[0] - 0)) - 1
        D1 = np.exp(abs(y[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i+2]-2*y[i+1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
        
        elif i==n-1:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i-2]-2*y[i-1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
        
        else:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i-1]+y[i+1]-2*y[i])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
    if j > 10e1:
        j=10e1
    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + 10*orto**2 + 10*orto1**2 + pen
    if exponential is True:
        Z = -np.exp(-cons*Z) #Función acotada entre -1 y 0. Óptimo en -1
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z


########## 3er  estado excitado  #####################
def oneDHOExpE3(y:Union[np.array, list], x:Union[np.array, list], E0:float, Psi0:Union[np.array, list],  E1:float, Psi1:Union[np.array, list], E2:float, Psi2:Union[np.array, list], exponential:bool = False) -> float:
    "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
    #Energy_lower bound
    E = E0+E1+E2 #results_dict['eda_resultExpE1'].sel_inds_hist[-1, 0, -1]**0.5 + results_dict['eda_resultExpE'].sel_inds_hist[-1,0,-1]**0.5 + results_dict['eda_resultExpE2'].sel_inds_hist[-1,0,-1]**0.5 #Exp E n=2
    #grid size x=[-4, 4]
    n = len(y)
    #x=np.linspace(-4, 4, n)
    dx = (x[-1]-x[0])/(n-1)
    #integral sums
    ExpE = 0; j=0; orto=0; orto1=0 ; orto2=0
    cons=10
    # if (y[0]< -0.1 or y[0] > 0.1) or (y[n-1]<-0.1 or y[n-1]>0.1):
    #     F = np.infty
    #     return F
    # else:

    ## Boundary condition as penalization term
    pen = 0
    if (y[0] < -1e-4 or y[0] >1e-4) or (y[-1] < -1e-4 or y[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(y[0] - 0)) - 1
        D1 = np.exp(abs(y[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i+2]-2*y[i+1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
            orto2 = orto2 + y[i]*Psi2[i]*dx
        
        elif i==n-1:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i]+y[i-2]-2*y[i-1])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
            orto2 = orto2 + y[i]*Psi2[i]*dx
        
        else:
            ExpE = ExpE + y[i]*dx*(-0.5*(y[i-1]+y[i+1]-2*y[i])/dx**2 + 0.5*x[i]**2*y[i])
            j = j + y[i]**2 * dx
            orto = orto + y[i]*Psi0[i]*dx
            orto1 = orto1 + y[i]*Psi1[i]*dx
            orto2 = orto2 + y[i]*Psi2[i]*dx
    if j > 10e1:
        j=10e1    
    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + 10*orto**2 + 10*orto1**2 + 10*orto2**2 + pen
    if exponential is True:
        Z = -np.exp(-cons*Z) #Función acotada entre -1 y 0. Óptimo en -1
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z