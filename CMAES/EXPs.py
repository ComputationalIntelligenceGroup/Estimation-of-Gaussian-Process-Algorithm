import numpy as np
import sys
import os
# Get the absolute path of the main directory
main_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add the main directory to sys.path
sys.path.append(main_dir)
from pymooOptFunctions.pymooOptFunctions import *
import time
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import pickle
#from pymoo.termination.default import MultiObjectiveTermination
from pymoo.termination import get_termination

def _normalizeCMAES(data:np.array, grid_size: float):
    '''
    Normalizes the solutions i.e. multiply by a constant so that the integral of the solution^2 = 1.
    Be aware that integration depends on problem definition and size!!!!
    '''
    #grid_size = L/variables     dx of integral
    #grid_size = 5/(self.n_variables-1)

    #Assuming a constant size grid we know grid size independently of problem parameters (better code-wise)
    #self.grid_size = grid_X[-1,0] - grid_X[-2, 0]
    integral = 0
    for var in range(data.shape[0]):
        integral = integral + (data[var]**2)*grid_size ##Tener en cuenta posibles cambios en tamaño grid!!! L caja = 5 

    A = (1/integral)**0.5
    data = A * data
    return data

#Medidas de los experimentos se guardan en un array.shape [Nexp, 9, 5]
#porque se harán Nexp ejecuciones de los problemas
# 9 filas porque tenemos 4 estados de Oscilador, 4 de pozo y 1 de NLSE
# 5 métricas [MSE, E, E%, GP_1000, t_exc] GP_1000 queda desierto en algoritmos diferentes 
# en este orden de manera que por ejemplo los E% de los Nexp del estado Psi0 del pozo es el elemento [:, 4, 2] del array de Resultados

n_variables = [800] #[50, 425, 800]Tamaños del grid (número de dimensiones del problema de optimización)
for vars in n_variables:
    Nexp = 10
    Resultados = np.empty((Nexp, 9, 5))

    ### Older stop criteria ###
    ###Limit of iterations for CMAES
    # if vars==50:
    #     maxgen = 8000
    # if vars==425:
    #     maxgen = 15000
    # if vars==800:
    #     maxgen = 20000
    # #termination = get_termination('soo', ftol=1e-1, period=100, n_max_gen=maxgen, n_max_evals=np.inf, xtol=1e-10) and get_termination("time", "00:15:00")
    # #termination = get_termination('soo', ftol=1e-8, period=60, n_max_gen=maxgen, n_max_evals=np.inf, xtol=1e-10) and get_termination("time", "00:00:50")
    # termination = get_termination('soo', ftol=1e-8, period=60, n_max_gen=maxgen, n_max_evals=np.inf, xtol=1e-10)
    # #get_termination('soo', ftol=1e-6, period=200, n_max_gen=8000) or get_termination("time", "00:00:10")#or get_termination("n_gen", 10000) or get_termination("time", "00:00:20")
    # #termination = get_termination('n_gen', 20000) get_termination("ftol", tol=1e-4, n_last=80)
    # #termination = MultiObjectiveTermination([get_termination("n_gen", 1000), get_termination("time", "00:15:00"), get_termination("soo", ftol=1e-6, period=50)])

    ###################################################################################
    ### STOP criteria for algorithm ###
    max_f_evals = 1*10**6 ##This is the main criteria to stop algorithm, max_f_evals = max_iter * size_gen.
    ###################################################################################


    print(f'\n\n########################### Experimentos con {vars} variables ###########################\n\n')
    for experimento in range(Nexp):
        print(f'\n\n########################### Ejecución de Experimentos número: {experimento+1} de {Nexp} ###########################\n\n')
        #################################################################################
        ############## Quantum Harmonic Oscillator 1D ###################################
        #################################################################################
        #First we load the hyperparameters for this problem and this number of variables        
        ruta = f'CMAESBO_params/{vars}Osc'
        with open(f'{ruta}.pkl', "rb") as file:
            BO_params = pickle.load(file)

        sigma = BO_params['sigma']
        pop_size = BO_params['pop_size']
        restarts = BO_params['restarts']
        bipop = BO_params['bipop']
        incpopsize = BO_params['incpopsize']

        ### STOP criteria for algorithm ###
        max_iter = int(max_f_evals/pop_size)
        dead_iter = int(max_iter*0.05) #dead_iter = 5% of max_iter
        termination = get_termination('soo', ftol=1e-10, period=dead_iter, n_max_gen=max_iter, n_max_evals=max_f_evals, xtol=1e-10)

        #Medidas de los experimentos se guardan en un array.shape [Nexp, 9, 5]
        #porque se harán Nexp ejecuciones de los problemas
        # 9 filas porque tenemos 4 estados de Oscilador, 4 de pozo y 1 de NLSE
        # 5 métricas [MSE, E, E%, GP_1000, t_exc]
        # en este orden de manera que por ejemplo los E% de los Nexp del estado Psi0 del pozo es el elemento [:, 4, 2] del array de #Resultados
        print('\nQuantum Harmonic Oscillator IN PROGRESS\n')
        #Energías teóricas
        Eg = 0.5
        En=[]
        for i in range(4):
            e= Eg + 1*i
            En.append(e)

        X_grid = np.linspace(start=-4, stop=4, num=vars).reshape(-1,1)
        grid_size = ((X_grid[-1]-X_grid[0])/(len(X_grid)-1))

        ########################################################
        #### Analytical solutions definitions ##################
        ########################################################
        #Ground state
        APsi0 = np.exp(-X_grid**2/2)
        integral =0
        for i in range(len(APsi0)):
            integral = integral + (APsi0[i]**2)*((X_grid[-1]-X_grid[0])/(len(APsi0)-1))

        B = (1/integral)**0.5
        APsi0 = B * APsi0

        #1st excitated state
        APsi1 = np.exp(-X_grid**2/2)*2*X_grid
        integral =0
        for i in range(len(APsi1)):
            integral = integral + (APsi1[i]**2)*((X_grid[-1]-X_grid[0])/(len(APsi1)-1))

        B = (1/integral)**0.5
        APsi1 = -1 * B * APsi1 #signo menos para correspondencia con solución algoritmo (simetrización)

        #2nd excitated state
        APsi2 = np.exp(-X_grid**2/2)*(4*X_grid**2-2)
        integral =0
        for i in range(len(APsi2)):
            integral = integral + (APsi2[i]**2)*((X_grid[-1]-X_grid[0])/(len(APsi2)-1))

        B = (1/integral)**0.5
        APsi2 = B * APsi2

        #if results_dict['eda_resultExpE2'].sel_inds_hist[-1, 0, :-1][3]<0:
        # APsi2=-APsi2

        #3rd excitated state
        APsi3 = np.exp(-X_grid**2/2)*(8*X_grid**3-12*X_grid)
        integral =0
        for i in range(len(APsi3)):
            integral = integral + (APsi3[i]**2)*((X_grid[-1]-X_grid[0])/(len(APsi3)-1))

        B = (1/integral)**0.5
        APsi3 = -1 * B * APsi3 #signo menos para correspondencia con solución algoritmo (simetrización)

        APsis = {}
        APsis['APsi0'] = APsi0[:, 0]
        APsis['APsi1'] = APsi1[:, 0]
        APsis['APsi2'] = APsi2[:, 0]
        APsis['APsi3'] = APsi3[:, 0]
        X_grid = X_grid[:, 0]

        ############################################################################
        ############################################################################

        #Ground state n=1
        print(f'F should be approx {En[0]**2}')

        problem = pymoooOscGround(n_var=vars, grid=X_grid, eps=1e-4)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resground = minimize(problem, algorithm, termination, verbose=True)#,seed=1,verbose=True)
        Resultados[experimento, 0, 4] = resground.exec_time # 5 métricas [MSE, E, E%, GP_1000, t_exc]

        Psi_fund=resground.X
        Psi_fund = _normalizeCMAES(Psi_fund, grid_size)
        E_fund = np.sqrt(resground.F[0]) # def función de coste la energía se determina a partir del fitness raiz cuadrada diff Ener
        Resultados[experimento, 0, 1] = E_fund
        Resultados[experimento, 0, 2] = abs(E_fund-En[0])/En[0]*100 
        params = {'E': E_fund, 'Psi': Psi_fund}
        print(f'\n  |Psi0> done {resground.F}\n')

        #1excitated state n=2
        print(f'F should be approx {(En[1]-En[0])**2}')
        problem = pymoooOsc1(n_var=vars, grid=X_grid, E0=E_fund, Psi0=Psi_fund, eps=1e-4)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#, x0=initial_guess[46, :])
        resOsc1 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 1, 4] = resOsc1.exec_time

        Psi_1=resOsc1.X
        Psi_1 = _normalizeCMAES(Psi_1, grid_size)
        E1 = np.sqrt((resOsc1.F[0]))
        Resultados[experimento, 1, 1] = E_fund + E1
        Resultados[experimento, 1, 2] = abs(E_fund + E1 -En[1])/En[1]*100
        params = {'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1}
        print(f'\n  |Psi1> done {resOsc1.F}\n')

        #2nd excitated state n=3
        print(f'F should be approx {(En[2]-En[1])**2}')
        problem = pymoooOsc2(n_var=vars,  grid=X_grid, eps=1e-4, E0=E_fund, Psi0=Psi_fund, E1=E1, Psi1=Psi_1)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resOsc2 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 2, 4] = resOsc2.exec_time
        Psi_2=resOsc2.X
        Psi_2 = _normalizeCMAES(Psi_2, grid_size)
        E2 = np.sqrt(resOsc2.F[0])
        Resultados[experimento, 2, 1] = E_fund + E1 + E2
        Resultados[experimento, 2, 2] = abs(E_fund + E1 +E2 -En[2])/En[2]*100

        params = {'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1, 'E2': E2, 'Psi2': Psi_2}
        print(f'\n  |Psi2> done {resOsc2.F}\n')

        #3rd excitated state n=4
        print(f'F should be approx {(En[3]-En[2])**2}') #-0.1482')
        problem = pymoooOsc3(n_var=vars,  grid=X_grid, eps=1e-4, E0=E_fund, Psi0=Psi_fund, E1=E1, Psi1=Psi_1, E2=E2, Psi2=Psi_2)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resOsc3 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 3, 4] = resOsc3.exec_time
        Psi_3 = resOsc3.X
        Psi_3 = _normalizeCMAES(Psi_3, grid_size)
        E3 = np.sqrt(resOsc3.F[0])
        Resultados[experimento, 3, 1] = E_fund + E1 + E2 + E3
        Resultados[experimento, 3, 2] = abs(E_fund + E1 +E2+E3 -En[3])/En[3]*100
        params = {'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1, 'E2': E2, 'Psi2': Psi_2, 'E3':E3, 'Psi3': Psi_3}
        print(f'\n  |Psi3> done {resOsc3.F}\n')

        #Cálculos de métricas error MSE de cada estado con estado teórico
        MSEs = []
        for i in range(4):
            mse = min(np.mean((params[f'Psi{i}'] - APsis[f'APsi{i}'])**2), np.mean((-1*params[f'Psi{i}'] - APsis[f'APsi{i}'])**2))
            Resultados[experimento, i, 0] = mse
            MSEs.append(mse)

        print('\nQuantum Harmonic Oscillator DONE\n')
            
        ######################################################################################
        ############## POZO DE POTENCIAL INFINITO 1D WELL ####################################
        ######################################################################################
        #First we load the hyperparameters for this problem and this number of variables        
        ruta = f'CMAESBO_params/{vars}Well'
        with open(f'{ruta}.pkl', "rb") as file:
            BO_params = pickle.load(file)

        sigma = BO_params['sigma']
        pop_size = BO_params['pop_size']
        restarts = BO_params['restarts']
        bipop = BO_params['bipop']
        incpopsize = BO_params['incpopsize']

        ### STOP criteria for algorithm ###
        max_iter = int(max_f_evals/pop_size)
        dead_iter = int(max_iter*0.05) #dead_iter = 5% of max_iter
        termination = get_termination('soo', ftol=1e-8, period=dead_iter, n_max_gen=max_iter, n_max_evals=max_f_evals, xtol=1e-10)

        print('\nInfinite Well IN PROGRESS\n')
        L=5
        #Energías teóricas
        Eg = (2*np.pi)**2/(8*L**2*1)
        En=[]
        for i in range(4):
            e=(i+1)**2*Eg
            En.append(e)

        grid = np.linspace(0, L, vars)
        grid_size = grid[-1]-grid[-2]

        APsisWell = {}
        for n in range(4):
            APsisWell[f'APsi{n}'] = (2/L)**0.5*np.sin((n+1)*np.pi*grid/L)

        #Ground state n=1
        print(f'F should be approx {En[0]**2}')
        problem = pymoooWellGround(n_var=vars, eps=1e-4)        
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resground = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 4, 4] = resground.exec_time # 5 métricas [MSE, E, E%, GP_1000, t_exc]

        Psi_fund=resground.X
        Psi_fund = _normalizeCMAES(Psi_fund, grid_size)
        E_fund = np.sqrt(resground.F[0]) # def función de coste la energía se determina a partir del fitness raiz cuadrada diff Ener
        Resultados[experimento, 4, 1] = E_fund
        Resultados[experimento, 4, 2] = abs(E_fund-En[0])/En[0]*100 
        params = {'L':L, 'E': E_fund, 'Psi': Psi_fund}
        print(f'\n  |Psi0> done {resground.F}\n')

        #1excitated state n=2
        print(f'F should be approx {(En[1]-En[0])**2}') #-0.7042')
        problem = pymooWell1(n_var=vars, eps=1e-4, E=E_fund, Psi=Psi_fund)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resWell1 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 5, 4] = resWell1.exec_time

        Psi_1=resWell1.X
        Psi_1 = _normalizeCMAES(Psi_1, grid_size)
        E1 = np.sqrt((resWell1.F[0]))
        Resultados[experimento, 5, 1] = E_fund + E1
        Resultados[experimento, 5, 2] = abs(E_fund + E1 -En[1])/En[1]*100
        params = {'L':L, 'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1}
        print(f'\n  |Psi1> done {resWell1.F}\n')

        #2nd excitated state n=3
        print(f'F should be approx {(En[2]-En[1])**2}') #-0.3775')
        problem = pymooWell2(n_var=vars, eps=1e-4, E0=E_fund, Psi0=Psi_fund, E1=E1, Psi1=Psi_1)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resWell2 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 6, 4] = resWell2.exec_time
        Psi_2=resWell2.X
        Psi_2 = _normalizeCMAES(Psi_2, grid_size)
        E2 = np.sqrt(resWell2.F[0])
        Resultados[experimento, 6, 1] = E_fund + E1 + E2
        Resultados[experimento, 6, 2] = abs(E_fund + E1 +E2 -En[2])/En[2]*100

        params = {'L':L, 'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1, 'E2': E2, 'Psi2': Psi_2}
        print(f'\n  |Psi2> done {resWell2.F}\n')

        #3rd excitated state n=4
        print(f'F should be approx {(En[3]-En[2])**2}') #-0.1482')
        problem = pymooWell3(n_var=vars, eps=1e-4, E0=E_fund, Psi0=Psi_fund, E1=E1, Psi1=Psi_1, E2=E2, Psi2=Psi_2)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=initial_guess)
        resWell3 = minimize(problem, algorithm, termination)#,seed=1,verbose=True)
        Resultados[experimento, 7, 4] = resWell3.exec_time
        Psi_3 = resWell3.X
        Psi_3 = _normalizeCMAES(Psi_3, grid_size)
        E3 = np.sqrt(resWell3.F[0])
        Resultados[experimento, 7, 1] = E_fund + E1 + E2 + E3
        Resultados[experimento, 7, 2] = abs(E_fund + E1 +E2+E3 -En[3])/En[3]*100
        params = {'L':L, 'E0': E_fund, 'Psi0': Psi_fund, 'E1':E1, 'Psi1': Psi_1, 'E2': E2, 'Psi2': Psi_2, 'E3':E3, 'Psi3': Psi_3}
        print(f'\n  |Psi3> done {resWell3.F}\n')

        #Cálculos de métricas error MSE de cada estado con estado teórico
        MSEs = []
        for i in range(4):
            #May converge to Psi or -Psi, thus
            mse = min(np.mean((params[f'Psi{i}'] - APsisWell[f'APsi{i}'])**2), np.mean((-1*params[f'Psi{i}'] - APsisWell[f'APsi{i}'])**2))
            Resultados[experimento, i+4, 0] = mse
            MSEs.append(mse)

        print('\nInfinite Well DONE\n')

        #########################################################
        ######### N L S E #######################################
        #########################################################
        print('\nNLSE IN PROGRESS\n')
        #First we load the hyperparameters for this problem and this number of variables
        rutafinal = f'CMAES_results/{vars}'
        if vars%2 ==0:
            varsNLSE=vars+1
        else:
            varsNLSE=vars

        ruta = f'CMAESBO_params/{varsNLSE}NLSE'
        with open(f'{ruta}.pkl', "rb") as file:
            BO_params = pickle.load(file)

        sigma = BO_params['sigma']
        pop_size = BO_params['pop_size']
        restarts = BO_params['restarts']
        bipop = BO_params['bipop']
        incpopsize = BO_params['incpopsize']

        ### STOP criteria for algorithm ###
        max_iter = int(max_f_evals/pop_size)
        dead_iter = int(max_iter*0.05) #dead_iter = 5% of max_iter
        termination = get_termination('soo', ftol=1e-10, period=dead_iter, n_max_gen=max_iter, n_max_evals=max_f_evals, xtol=1e-10)

        problem = pymooNLSE(n_var=varsNLSE, eps=1e-4)
        algorithm = CMAES(sigma=sigma, pop_size=pop_size, restarts=restarts, bipop=bipop, incpopsize=incpopsize)#x0=np.ones(nvar))
        CMAESres = minimize(problem, algorithm, termination)
        Resultados[experimento, 8, 4] = CMAESres.exec_time # 5 métricas [MSE, E, E%, GP_1000, t_exc]
        MSE = np.sum((1/np.cosh(np.linspace(-1, 1, varsNLSE)) - CMAESres.X)**2) / varsNLSE
        Resultados[experimento, 8, 0] = MSE
        #setting non used columns to None
        Resultados[experimento, 8, 1:3] = None
        Resultados[experimento, :, 3] = None
        print('\nNLSE DONE\n')
        print(f'EXPERIMENT {experimento+1} of {Nexp} FINISHED\n')
    #Finally results is saved with pickle. ruta is defined previous modification of vars if is even
    #rutafinal = f'CMAES_results/{vars}'
    with open(f'{rutafinal}.pkl', "wb") as file:
        pickle.dump(Resultados, file)
