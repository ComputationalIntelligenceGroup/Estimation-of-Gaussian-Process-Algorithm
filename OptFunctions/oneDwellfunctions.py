import numpy as np
from numpy.linalg import norm
from typing import Union

#Definición de la función de optimización basada en referencia: Direct solution of Schrödinger equation by genetic algorithm:
#test cases
#########FUNDAMENTAL STATE######################
def oneD_wellExpE(x:Union[np.array, list],  L:float, exponential:bool = False) -> float:
    #Energy_lower bound
    E = 0
    #Well lenght, by default is set equal to 5
    #L=5
    n = len(x)
    ExpE = 0; j=0
    #cons=1*10**(-6)
    # if (x[0]< -0.01 or x[0] > 0.01) or (x[n-1]<-0.01 or x[n-1]>0.01):
    #     F = np.inf
    #     return F
    
    ## Boundary condition as penalization term
    pen = 0
    if (x[0] < -1e-4 or x[0] >1e-4) or (x[-1] < -1e-4 or x[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(x[0] - 0)) - 1
        D1 = np.exp(abs(x[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
            j = j + x[i]**2* (L/(n-1))

        elif i==n-1:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
            j = j + x[i]**2 * (L/(n-1))
            
        else:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
            j=j+x[i]**2* (L/(n-1))


    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4
            
    Z = (ExpE/j - E)**2 + pen

    if exponential is True:
        Z = -np.exp(-Z) #Función acotada entre -1 y 0. Óptimo en -1
    # #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z

# def oneD_wellExpEexponential(x:Union[np.array, list]) -> float:
#     #Energy_lower bound
#     E = 0
#     #Well lenght
#     L=5
#     n = len(x)
#     ExpE = 0; j=0
#     cons=1*10**(-6)
#     if (x[0]< -0.01 or x[0] > 0.01) or (x[n-1]<-0.01 or x[n-1]>0.01):
#         F = np.inf
#         return F
    
#     else:
#         for i in range(n):
#             if i==0:
#                 ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
#                 j = j + x[i]**2* (L/(n-1))

#             elif i==n-1:
#                 ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
#                 j = j + x[i]**2 * (L/(n-1))
                
#             else:
#                 ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
#                 j=j+x[i]**2* (L/(n-1))
            
#     Z = (ExpE/j - E)**2
#     Z = -np.exp(-Z) #Función acotada entre -1 y 0. Óptimo en -1
#     #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
#     return Z



######### 1st EXCITATED STATE ##################
#Definición de la función de optimización basada en referencia: Direct solution of Schrödinger equation by genetic algorithm:
#test cases
###############################
def oneD_wellExpEexcitated(x:Union[np.array, list],  L:float, E:float, Psi:np.array, exponential: bool = False) -> float:
    '''E is the energy of the previously calculated ground state. In this case it corresponds to the square root of the function evaluation.
    ¡¡¡¡¡¡¡¡¡¡¡¡¡ The previous statmente Won't be valid if using exponential form of cost function at any point of the execution !!!!!!!!!!!!!!!!!!!!!!!!!!
     Psi is an array containing the amplitudes for the fixed grid points '''
    #Energy_lower bound
    #E = results_dict['eda_resultExpE'].sel_inds_hist[-1, 0, -1]**0.5
    #Psi = results_dict['eda_resultExpE'].sel_inds_hist[-1,0,:-1] in original code

    #Well lenght
    #L=5
    n = len(x)
    ExpE = 0; j=0 #Variables to store Expected energy of current solution and norm of current solution
    #Ortogonal term defs
    beta = 10
    orto=0
    #cons=1*10**(-6)
    # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
    #     F = np.inf
    #     return F
    

    ## Boundary condition as penalization term
    pen = 0
    if (x[0] < -1e-4 or x[0] >1e-4) or (x[-1] < -1e-4 or x[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(x[0] - 0)) - 1
        D1 = np.exp(abs(x[-1] - 0)) - 1
        pen = D + D1
    
    
    for i in range(n):
        if i==0:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
            j = j + x[i]**2* (L/(n-1))
            orto = orto + x[i]*Psi[i]*(L/(n-1))


        elif i==n-1:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
            j = j + x[i]**2 * (L/(n-1))
            orto = orto + x[i]*Psi[i]*(L/(n-1))
            
        else:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
            j=j+x[i]**2* (L/(n-1))
            orto = orto + x[i]*Psi[i]*(L/(n-1))

    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4
            
    Z = (ExpE/j - E)**2 + beta* orto**2 + pen
    if exponential is True:
        Z = -np.exp(-Z)
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z



######### 2nd EXCITATED STATE #################################

def oneD_wellExpEexcitated3(x:Union[np.array, list],  L:float, E0:float, E1: float, Psi0:np.array, Psi1:np.array, exponential: bool = False) -> float:
    '''E0 E1 are the energies of the previously calculated states. If not using exponential shape for fitness function energies correspond to the square
    root of the actual fitness evaluation. Supposing that ortogonal term is 0 the erergy of the previous state is again simply the square root of fitness
     evaluation. Energy lower bound in this case is simply the sum of the previous energies. (Sums of squares roots of previous energy values) '''
    #Energy_lower bound
    #E = (results_dict['eda_resultExpExcitated'].sel_inds_hist[-1, 0, -1]**0.5 + results_dict['eda_resultExpE'].sel_inds_hist[-1, 0, -1]**0.5)
    E = E0 + E1
    #Previous states
    #Psi0 = results_dict['eda_resultExpE'].sel_inds_hist[-1,0,:-1]
    #Psi1 = results_dict['eda_resultExpExcitated'].sel_inds_hist[-1,0,:-1]

    #Well lenght
    #L=5
    n = len(x)
    ExpE = 0; j=0
    #Ortogonal term defs
    beta = 10
    orto=0; orto2 = 0
    #cons=1*10**(-6)
    # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
    #     F = np.inf
    #     return F
    
    ## Boundary condition as penalization term
    pen = 0
    if (x[0] < -1e-4 or x[0] >1e-4) or (x[-1] < -1e-4 or x[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(x[0] - 0)) - 1
        D1 = np.exp(abs(x[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
            j = j + x[i]**2* (L/(n-1))
            orto = orto + x[i]*Psi1[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi0[i]*(L/(n-1))


        elif i==n-1:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
            j = j + x[i]**2 * (L/(n-1))
            orto = orto + x[i]*Psi1[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi0[i]*(L/(n-1))
            
        else:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
            j=j+x[i]**2* (L/(n-1))
            orto = orto + x[i]*Psi1[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi0[i]*(L/(n-1))

    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + beta* orto**2 + beta * orto2**2 + pen

    if exponential is True:
        Z = -np.exp(-Z)
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar
    return Z

####### 3rd EXCITATED STATE #########################

def oneD_wellExpEexcitated4(x:Union[np.array, list], L:float, E0:float, E1: float, E2:float, Psi0:np.array, Psi1:np.array, Psi2:np.array, exponential: bool = False) -> float:
    #Energy_lower bound
    #E = (results_dict['eda_resultExpExcitated'].sel_inds_hist[-1, 0, -1]**0.5 + results_dict['eda_resultExpE'].sel_inds_hist[-1, 0, -1]**0.5 +results_dict['eda_resultExpExcitated3'].sel_inds_hist[-1,0,-1]**0.5)
    E = E0+E1+E2
    #Previous states
    #results_dict['eda_resultExpExcitated3'].sel_inds_hist[-1,0,:-1]
    #results_dict['eda_resultExpE'].sel_inds_hist[-1,0,:-1]
    #results_dict['eda_resultExpExcitated'].sel_inds_hist[-1,0,:-1]

    #Well lenght
    #L=5
    n = len(x)
    ExpE = 0; j=0
    #Ortogonal term defs
    beta = 10
    orto=0; orto2 = 0; orto3=0
    #cons=1*10**(-6)
    # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
    #     F = np.inf
    #     return F
    
    
    ## Boundary condition as penalization term
    pen = 0
    if (x[0] < -1e-4 or x[0] >1e-4) or (x[-1] < -1e-4 or x[-1] >1e-4):
        #pen = 10e3
        D = np.exp(abs(x[0] - 0)) - 1
        D1 = np.exp(abs(x[-1] - 0)) - 1
        pen = D + D1

    for i in range(n):
        if i==0:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
            j = j + x[i]**2* (L/(n-1))
            orto3 = orto3 + x[i]*Psi2[i]*(L/(n-1))
            orto = orto + x[i]*Psi0[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi1[i]*(L/(n-1))


        elif i==n-1:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
            j = j + x[i]**2 * (L/(n-1))
            orto3 = orto3 + x[i]*Psi2[i]*(L/(n-1))
            orto = orto + x[i]*Psi0[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi1[i]*(L/(n-1))
            
        else:
            ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
            j=j+x[i]**2* (L/(n-1))
            orto3 = orto3 + x[i]*Psi2[i]*(L/(n-1))
            orto = orto + x[i]*Psi0[i]*(L/(n-1))
            orto2 = orto2 + x[i]*Psi1[i]*(L/(n-1))

    #Avoiding trivial solution. Enforcing norm of state to be 1
    D2 = np.exp(abs(j - 1)) - 1
    pen = pen + D2

    if j <= 0 + 1e-4:
        j = j + 1e-4

    Z = (ExpE/j - E)**2 + beta* orto**2 + beta * orto2**2 + beta* orto3**2 + pen
    #F = -1*np.exp(-cons*Z) #con signo menos global para minimizar

    if exponential is True:
        Z = -np.exp(-Z)
    return Z


###### FUNCTIONS/SOLUTIONS NORMALIZATION #########
## The integral of the square of solutions (wave function) must be equal to one
# def Normalization(results_dict: dict) -> dict:
#     '''Given a dictionary of edaresults objects this function returns a new dictionary with same keys but with normalized solutions'''

#     Nresults_dict = {}
#     for key in results_dict.keys():
#         shape = (len(results_dict[key].sel_inds_hist), len(results_dict[key].sel_inds_hist[0,:,0]), len(results_dict[key].sel_inds_hist[-1,0,:-1])+1)
#         Nresult = np.empty(shape)
#         for gen in range(len(results_dict[key].sel_inds_hist)):
#             for ind in range(len(results_dict[key].sel_inds_hist[0,:,0])):
#                 integral = 0
#                 for var in range(len(results_dict[key].sel_inds_hist[-1,0,:-1])):
#                     integral = integral + (results_dict[key].sel_inds_hist[gen, ind,:-1][var]**2)*(5/(n_variables-1)) ###Tener en cuenta posibles cambios en tamaño grid!!! L caja = 5 

#                 A = (1/integral)**0.5
#                 Nresult[gen, ind, :-1] = A * results_dict[key].sel_inds_hist[gen, ind, :-1]
#                 Nresult[gen, ind, -1] = results_dict[key].sel_inds_hist[gen, ind, -1]

#         Nresults_dict[f'N{key}'] = Nresult
