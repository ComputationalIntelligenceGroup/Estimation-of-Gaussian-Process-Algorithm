import numpy as np
from scipy.stats import norm
from utils import _parallel_apply_along_axis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from time import process_time
import copy


class EGPAResult:

    """
    Object used to encapsulate the result and information of the EDA during the execution
    """

    def __init__(self,
                 best_ind: np.array,
                 best_cost: float,
                 n_fev: int,
                 best_sampled: np.array,
                 data: np.array,
                 means: list,
                 stds: list,
                 GP_hist: list,
                 cpu_time: float):

        """

        :param best_ind: Best result found in the execution.
        :param best_cost: Cost of the best result found.
        :param n_fev: Number of cost function evaluations.
        :param best_sampled: 3D array (gen, bestsampled ind, variables+fitness)
        :param data: 3D array (gen, sorted with fitness inds, variables+fitness) with all generated individuals
        :param means: list of lists with means in each grid position in a given iteration
        :param stds: list of lists with stds in each grid position in a given iteration
        :param GP_hist: list with Gaussian Process object from each generation
        :param cpu_time: CPU time invested in the optimization.
        """

        self.best_ind = best_ind
        self.best_cost = best_cost
        self.n_fev = n_fev
        self.best_sampled = best_sampled
        self.data = data
        self.means = means
        self.std = stds
        self.GP_hist = GP_hist
        self.cpu_time = cpu_time

    def __str__(self):
        string = "\tNFVALS = " + str(self.n_fev) + " F = " + str(self.best_cost) + "CPU time (s) = " + \
                 str(self.cpu_time)
        return string
    




## ESTIMATION OF GAUSSIAN PROCESS ALGORITHM ##



class EGPA():

    """
    This class implements the Estimation of Gaussian Process Algortihm. The goal is to obtain a gaussian process where the mean corresponds with
    some specific points observations f(x) that are refined iteratively. For a fixed set of grid points Gaussian Process is started (with optional initial
    observations). Functions are sampled from the GP (for fixed grid points x and observations f(x)) and the best ones, according to a optimization function,
    are selected. The mean and std of f(x) at each observation point x is calculated from selected functions and this is used to fit a new Gaussian Process
    with these evidences with noise at each grid point. 
    """
    def __init__(self,
                 size_gen: int,
                 grid_X: np.array,
                 max_iter: int,
                 dead_iter: int,
                 alpha: float = 0.5,
                 elite_factor: float = 0.4,
                 init_kernels: int = 10,
                 std_lower_bound: float = 0.2,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: tuple = None,
                 normalize_solutions: bool = True,
                 symmetrize_solutions: bool = True,
                 analysis_mode: bool = False):


        # super().__init__(size_gen=size_gen, grid_X=grid_X ,max_iter=max_iter, dead_iter=dead_iter, #n_variables=n_variables,
        #                  alpha=alpha, elite_factor=elite_factor, disp=disp, parallelize=parallelize,
        #                  init_data=init_data)
        """
        :param size_gen: Number of functions to be sampled from GP in each iteration
        :param grid_X: array of shape (#of_grid_points, 1) with x fixed values
        :param max_iter: Number of max iterations of EGPA
        :param dead_iter: Number of max iterations without improvment after which algorithm will stop
        :param alpha: Proportion (respect to total size_gen solutions) of solutions to be used to re-learn and updated fit GP
        :param elite_factor: Proportion (respect to total size_gen solutions) of best-ever found solutions that must be used to compare and truncate solutions
        :param init_kernels: Number of different RBF kernels to initialize population
        :param std_lower_bound: Minumum value for standard deviation in each grid point observation. Value added to diagonal of covariance matrix
        :param disp: boolean to choose whether the algorithm run info should be displayed or not when minimize
        :param parallelize: boolean to choose whether the evaluation of solutions will be parallelize
        :param init_data: tuple of tuples with initial known observations or constraints 
               ((grid points x ), (observations f(x))) = ((x_1, x_2, ..., x_m), (f(x_1), f(x_2), ..., f(x_m)))
               both arrays must be shape ((n, 1)) where n=grid points. Observations must coincide with some of the points defined in the grid_X
        :param normalize_solutions: boolean to choose whether solutions will be normalize after sampling so integral of solution^2 = 1
        :param symmetrize_solutions: boolean to choose whether solutions are symmetrize (*-1) according to sign of function in some particular grid points
        :param analysis_mode: boolean to choose whether exhaustive info about solutions and GP from all generations is stored during the execution
        """

        self.size_gen = size_gen
        self.grid_X = grid_X
        self.max_iter = max_iter
        self.dead_iter = dead_iter
        self.alpha = alpha
        self.elite_factor = elite_factor
        self.init_kernels = init_kernels
        self.lower_bound = std_lower_bound #lower bound for std of observations at given point f(x) #0.01 para SE en potencial armónico y potencial caja
        self.disp = disp
        self.parallelize = parallelize
        self.init_data = init_data
        self.normalize_solutions = normalize_solutions
        self.symmetrize = symmetrize_solutions
        self.analysis_mode = analysis_mode

        self.truncation_length = int(size_gen * alpha)
        self.elite_length = int(size_gen * elite_factor)
        self.iteration = 1

        self.n_variables = grid_X.shape[0]
        self.grid_size = grid_X[-1,0] - grid_X[-2, 0]
        self.problem_size = grid_X[-1,0] - grid_X[0,0]
        # self.evaluations = np.array(0)
        # self.evaluations_elite = np.array(0)

        self.generation = None
        self.elite_temp = None
        self.pm = np.zeros((2, self.n_variables))

        if self.analysis_mode is True:
        #Vectores, lista de listas, para guardar las medias en cada punto del grid y su error
            self.pm_means = [] 
            self.pm_stds = []
        else:
            self.pm_means = None
            self.pm_stds = None

        #self.data = np.empty((1, size_gen, self.n_variables+1)) #la columna extra será el fitness

        ########### K E R N E L   del  PROCESO GAUSSIANO ################################
        ### Kernel is defined in learn function using hyperparameters of previous iterations to speed the convergence of hyperparams optimization.
        # self.lower_bound = 0.01 #lower bound for std of observations at given point f(x) #0.01 para SE en potencial armónico y potencial caja
        # self.kernel = 0.01 * RBF(length_scale=self.grid_size/np.sqrt(2), length_scale_bounds=(self.grid_size/np.sqrt(2), self.problem_size/np.sqrt(2))) #upper bound problem_size/2 para potencial armónico y caja
        # # self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
        #################################################################################

    def _initialize_population(self):
        '''
        Creates the initial population from the desired number of distinct RBF kernels. A 'grid search' is performed over the length scale parameter.
        The minimum value for the length scale is defined depending on the grid size, as lambda = grid_size/sqrt(2). 
        The maximum value is defined as half the total range of the independent variable.
        '''
                
        lengths = np.linspace(self.grid_size/np.sqrt(2), self.problem_size/2, self.init_kernels)
        samplesperkernel = int(self.size_gen/self.init_kernels)
        for scale in lengths:
            kernel = 0.01 * RBF(length_scale=scale,  length_scale_bounds='fixed') #length_scale_bounds=(scale[0], scale[0]))
            gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            gaussian_process.fit(self.init_data[0], self.init_data[1])

            if scale==lengths[0]:
                self.generation=gaussian_process.sample_y(self.grid_X, n_samples=samplesperkernel).T
            else:
                self.generation = np.concatenate([self.generation, gaussian_process.sample_y(self.grid_X, n_samples=samplesperkernel).T])


    def _sample_new_gen(self):
        """
        Samples new functions from the Gaussian Process.
        :param size: number of samplings from GP.
        :return: array with the dataset sampled. Fixed x grid points and observations f(x) sampled from GP
        :rtype: np.array
        """

        self.generation = self.gaussian_process.sample_y(self.grid_X, n_samples=self.size_gen).T
        # The mean of the Gaussian Process is also added to the population
        mean = self.gaussian_process.predict(self.grid_X) #mean has shape (grid_points, )
        mean = np.reshape(mean, (1, len(mean)))#reshaped mean (1, grid_points)
        self.generation = np.concatenate([self.generation, mean])

    def _normalize(self, dataset:np.array):
        '''
        Normalizes the solutions i.e. multiply by a constant so that the integral of the solution^2 = 1.
        Be aware that integration depends on problem definition and size!!!!
        '''
        #grid_size = L/variables     dx of integral
        #grid_size = 5/(self.n_variables-1)

        #Assuming a constant size grid we know grid size independently of problem parameters (better code-wise)
        #self.grid_size = grid_X[-1,0] - grid_X[-2, 0]

        for ind in range(len(dataset)):
                integral = 0
                for var in range(dataset.shape[1]):
                    integral = integral + (dataset[ind,:][var]**2)*self.grid_size ##Tener en cuenta posibles cambios en tamaño grid!!! L caja = 5 

                A = (1/integral)**0.5
                dataset[ind, :] = A * dataset[ind, :]

    # def _symmetrize(self, dataset:np.array):
    #     '''
    #     Symmetrize the solutions. In our problem both f(x) and -f(x) solve the equation and minimize the cost function.
    #     Como solución provisional a este problema identificamos el signo de las amplitudes en los primeros puntos del grid y las soluciones que no coincidan se multiplican por -1.
    #     Bastará con hacerlo una generación para los individuos seleccionados.
    #     '''
    #     print(f'Función de simetrización it {self.iteration} \n\n', 'Referencia de signo f(x_1) mejor:', dataset[0, 1], '\n')
    #     for row in range(1, len(dataset)):
    #         print('Antes de simetrización:', dataset[row, 1])
    #         if (dataset[0, 1] > 0 and dataset[row, 1] < 0):
    #             dataset[row, :] = dataset[row, :] * -1 
    #         if (dataset[0, 1] < 0 and dataset[row, 1] > 0):
    #             dataset[row, :] = dataset[row, :] * -1
    #         print('       Después de simetrización:', dataset[row, 1], '\n\n\n')
    

    def _learn(self, dataset: np.array):
        """
        Learns the mean and variances from selected (promising) observations f(x) for given and fixed x grid points. Then updates (re-learns) a new Gaussian
        Process, with the means f(x) as observed points with corresponding noise (Variances at each f(x)).

        :param dataset: dataset from which learn the probabilistic model. Will correspond to selected function observations.
        """
        for i in range(self.n_variables):
            self.pm[0, i] = np.mean(dataset[:, i])
            self.pm[1, i] = np.std(dataset[:, i])

            if self.pm[1, i] < self.lower_bound:
                self.pm[1, i] = self.lower_bound #lower bound of std for observation at given point f(x_i)

        if self.init_data is not None:
            #Fijamos las evidencias y condiciones de contorno acorde a las observaciones o condiciones de contorno conocidas
            for i in range(len(self.init_data[1])):
                self.pm[0, np.where(self.grid_X==self.init_data[0][i])[0]] = self.init_data[1][i]
                self.pm[1, np.where(self.grid_X==self.init_data[0][i])[0]] = 0

        if self.analysis_mode is True:    
            self.pm_means.append(self.pm[0,:])
            self.pm_stds.append(self.pm[1, :])

        #Ahora tenemos que actulizar el kernel, pasando como ruido a cada punto del grid estas std y como evidencias la media de amplitudes de las mejores sols
        #Definimos el Kernel inicial. Y ajustamos los datos obervados.
        if self.iteration >= 3:
            l = self.gaussian_process.kernel_.get_params()['k2'].length_scale
            self.kernel = 0.01 * RBF(length_scale=l, length_scale_bounds=(self.grid_size/np.sqrt(2), self.problem_size/np.sqrt(2)))
            self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=1, alpha=self.pm[1,:]**2) 
        else:
            self.kernel = 0.01 * RBF(length_scale=self.grid_size/np.sqrt(2), length_scale_bounds=(self.grid_size/np.sqrt(2), self.problem_size/2))
            self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, alpha=self.pm[1,:]**2)  

        #self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9, alpha=self.pm[1,:]**2)
        self.gaussian_process.fit(self.grid_X, self.pm[0,:])


    def _check_generation_parallel(self, objective_function: callable, params:dict , exp_from:int):

        if self.iteration >= exp_from:
            params['exponential'] = True 
            self.evaluations = _parallel_apply_along_axis(objective_function, 1, self.generation, **params)

        else:
            self.evaluations = _parallel_apply_along_axis(objective_function, 1, self.generation, **params)

        self.iteration += 1

    def _truncate(self):
        """ 
        Selection of the best individuals functions of the actual generation, comparing them against best ever found ones as well.
        """

        self.genpluseliteinds = np.concatenate([self.generation, self.elite_temp])
        self.genpluselitefitness =  np.append(self.evaluations, self.evaluations_elite)


        # first add the elite selection to be considered
        self.generation = np.concatenate([self.generation, self.elite_temp])
        self.evaluations = np.append(self.evaluations, self.evaluations_elite)
        

        # now we truncate
        self.ordering = self.evaluations.argsort()
        best_indices_truc = self.ordering[: self.truncation_length]
        best_indices_elit = self.ordering[: self.elite_length]
        self.elite_temp = self.generation[best_indices_elit, :]
        self.generation = self.generation[best_indices_truc, :]
        self.evaluations_elite = np.take(self.evaluations, best_indices_elit)
        self.evaluations = np.take(self.evaluations, best_indices_truc)

    def minimize(self, cost_function: callable, params: dict, exp_from: int ,output_runtime: bool = True):
        """
        Minimize function to execute the EGPA optimization. By default, the optimizer is designed to minimize a cost
        function; if maximization is desired, just add a minus sign to your cost function.

        :param cost_function: cost function to be optimized and accepts an array as argument.
        :param params: dict with **args to be passed to the cost function
        :exp_from: iteration number from which to use the exponential definition of cost function
        :param output_runtime: true if information during runtime is desired.
        :return: EdaResult object with results and information.
        :rtype: EdaResult
        """

        t1 = process_time()
        not_better = 0

        if self.analysis_mode is True:
        ### list to store pm object in each iteration
            GP_hist = []
        #####################################################
        
        for gen in range(0, self.max_iter):

            # if gen==0 and self.init_data is not None: #Provisional, no sé cuáles son los GP prior si no se pasan observaciones iniciales
            #     #self.kernel = 0.01 * RBF(length_scale=self.grid_size, length_scale_bounds=(self.grid_size, self.problem_size))
            #     self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
            #     self.gaussian_process.fit(self.init_data[0], self.init_data[1])

            if gen==0:
                self._initialize_population()
            else:
                self._sample_new_gen()

            if self.normalize_solutions:
                self._normalize(self.generation)

            # ### Antes simetrizaba fijándome en índices concretos fijados a mano
            # ### después se cambió según los valores del grid
            # ### 'Corregimos' las soluciones cuyas primeras amplitudes sean negativas.
            # ### de esta manera eliminamos el problema de la simetría de las soluciones.
           
            # for i in range(len(self.generation)):
            #     if (self.generation[i, 50] < 0) & (self.generation[i, 55] < 0) & (self.generation[i, 60] < 0) & (self.generation[i, 65] < 0) :
            #         self.generation[i, :] = self.generation[i, :]*-1

            #########################################################################################
            ############# S I M E T R I Z A C I Ó N    DE    FUNCIONES ##############################
            #########################################################################################
            #En función de los valores que puede tomar el grid (relacionado con el potencial concreto que estamos resolviendo)
            #fijamos unos índices u otros para corregir signos de las funciones
            #Esto se podría hacer también dentro de las funciones de coste castigando soluciones que no cumplan.
            if self.symmetrize is True:
                # Determinar si el grid tiene valores negativos
                if np.any(self.grid_X < 0): #Caso de potencial armónico y NLSE
                    # Intervalo relevante: [-2.2, -1.6]
                    mask = (self.grid_X >= -2.2) & (self.grid_X <= -1.6)
                    #print(mask.shape)
                else: #Caso de potencial de caja 1D 
                    # Intervalo relevante: [0.2, 1.1] 
                    mask = (self.grid_X >= 0.2) & (self.grid_X <= 1.1)
                
                # Obtener los índices del grid que cumplen la condición
                indices = np.where(mask)[0]
                #print(indices)
                
                # Verificar la condición sobre la función y cambiar el signo si es necesario
                for i in range(len(self.generation)):
                    if np.all(self.generation[i, indices] < 0):
                        self.generation[i, :] *= -1
            #########################################################################################
            #########################################################################################

            self._check_generation_parallel(cost_function, params, exp_from)
            best_fitness_local = min(self.evaluations)
            
            #### Guardar mejor individuo sampleado en un array 3D ###################################
            best_ind_local = np.where(self.evaluations == best_fitness_local)[0][0]
            best_ind_local = self.generation[best_ind_local]

            if gen==0:
                self.best_fitness_global = best_fitness_local
                self.best_ind_global = best_ind_local

            bestindplusfitness = np.hstack((best_ind_local, best_fitness_local))
            bestindplusfitness = np.reshape(bestindplusfitness, (1, bestindplusfitness.shape[0]))
            bestindplusfitness = np.reshape(bestindplusfitness, (1, bestindplusfitness.shape[0], bestindplusfitness.shape[1]))

            if gen==0:
                ### array to save best sampled individual (gen, variablesfitness)
                self.best_sampled = bestindplusfitness

            if gen > 0:
                self.best_sampled = np.vstack((self.best_sampled, bestindplusfitness))
            #############################################################################


            if gen==0:
                worst = max(self.evaluations)
                worst_ind = np.where(self.evaluations == worst)[0][0]
                self.elite_temp = np.reshape(self.generation[worst_ind], (1,len(self.generation[worst_ind])))
                self.evaluations_elite = worst
                
            self._truncate()

            #self._symmetrize(self.generation)
            #self._symmetrize(self.elite_temp)

            ########################################################################################################
            # 3D array (generation, individuals, variables + evaluation) creation
            if self.analysis_mode is True:   
            # Saving selected individuals only
                datagen = np.column_stack((self.genpluseliteinds[self.ordering[:self.size_gen], :], self.genpluselitefitness[self.ordering[:self.size_gen]])) #Once truncated we paste only size gen first indiviudals + evaluation
            ##### This way we are losing information of the last #elitelenght individuals, because we paste to the size_gen individuals the elite individuals we evaluate and sort and are only saving in data the first #size_gen ones
            if self.analysis_mode is False:
                datagen = np.column_stack((self.genpluseliteinds[self.ordering[:1], :], self.genpluselitefitness[self.ordering[:1]])) #Once truncated we paste only first indiviudal + evaluation
            
            if gen==0:
                self.data = np.reshape(datagen, (1, datagen.shape[0], datagen.shape[1]))
            #Guardamos todos los individuos, los seleccionados serán los inds_sel primeros
            if gen > 0:
                self.data = np.vstack((self.data, np.reshape(datagen, (1, datagen.shape[0], datagen.shape[1]))))
            #########################################################################################################


            ### Saving Gaussian Process object using deepcopy
            if gen>0:
                if self.analysis_mode is True:
                    pm_temp = copy.deepcopy(self.gaussian_process)
                    GP_hist.append(pm_temp)
                if self.analysis_mode is False:
                    GP_hist = copy.deepcopy(self.gaussian_process)

            self._learn(dataset=self.generation)

            # update the best values ever
            if best_fitness_local < self.best_fitness_global: #(< self.best_fitness_global + - tolerancia) se podría añadir
                self.best_fitness_global = best_fitness_local
                self.best_ind_global = best_ind_local
                not_better = 0

            else:
                not_better += 1
                if not_better == self.dead_iter:
                    break

            if output_runtime:
                print('IT: ', gen, '\tBest cost: ', self.best_fitness_global)

        if self.disp:
            print("\tNFEVALS = " + str((gen+1) * self.size_gen) + " F = " + str(self.best_fitness_global))
            print("\tX = " + str(self.best_ind_global))

        t2 = process_time()

        ###########################################################
        ####################Modified by Daniel Conde ##############
        ##### added best_ind_hist, sel_inds_hist, all_inds_hist, prob_mod_hist, prob_clone_hist to eda_result >>> requires changes in eda_result.py #######
        egpa_result = EGPAResult(self.best_ind_global, self.best_fitness_global, gen*self.size_gen,
                               self.best_sampled, self.data, self.pm_means, self.pm_stds ,GP_hist, t2-t1)

        return egpa_result


