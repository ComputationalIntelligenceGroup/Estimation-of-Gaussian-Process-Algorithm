import numpy as np
from pymoo.core.problem import ElementwiseProblem

### Problem implementations pymoo wise


####################################################
#####   N  L  S  E  ################################
####################################################
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

class pymooNLSE(ElementwiseProblem):

    def __init__(self, n_var, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=np.zeros(n_var),
                         xu= 1.1*np.ones(n_var))
        self.eps = eps

    def _evaluate(self, x, out, *args, **kwargs):
        L=2 #problem size from -1 to 1
        n = len(x)
        R = 0
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
        fd = (x[n//2+1]-x[n//2-1])/(2*h)
        D = np.exp(abs(fd - 0)) - 1

        ## Boundary condition as penalization term
        pen = 0
        if (x[n//2] < 1 - self.eps or x[n//2] > 1 + self.eps):
            #pen = 10e3
            pen = np.exp(abs(x[n//2] - 1)) - 1

        ## Boundary condition y(0) = 1 as two inequality conditions
        ## Approach |x - target| <= epsilon
        # ## x - target - eps <= 0   and    -x + target + eps <= 0 
        # g1 = x[n//2] - 1.00001 #This must be <= 0
        # g2 = -x[n//2] + 0.99999 #This must be <= 0

        # g1 = x[n//2] - 1 - self.eps #This must be <= 0
        # g2 = -x[n//2] + 1 -self.eps #This must be <= 0

        # out["F"] = 100*R + 10000*D + pen #[f1, f2]
        out["F"] = 100*R + 1000*D + 1000*pen
        #out["G"] =  [g1, g2]


#######################################################################################################
#####  O N E   D I M E N S I O N A L   I N F I N I N I T E   W E L L   ################################
#######################################################################################################
#########FUNDAMENTAL STATE######################

class pymoooWellGround(ElementwiseProblem):

    def __init__(self, n_var, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps

    def _evaluate(self, x, out, *args, **kwargs):
        #Energy_lower bound
        E = 0
        #Well lenght, by default is set equal to 5
        L=5
        n = len(x)
        ExpE = 0; j=0
        # #cons=1*10**(-6)
        # if (x[0]< -0.01 or x[0] > 0.01) or (x[n-1]<-0.01 or x[n-1]>0.01):
        #     F = np.inf
        #     return F
        
        # else:
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

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + pen

        ## Boundary conditions Psi(0) = Psi(n) = 0 as two inequality conditions per constraint
        ## Approach |x - target| <= epsilon
        ## x - target - eps <= 0   and    -x + target + eps <= 0
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]

######### 1st EXCITATED STATE ##################
class pymooWell1(ElementwiseProblem):
    
    def __init__(self, n_var, E, Psi,eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.E= E
        self.Psi = Psi

    def _evaluate(self, x, out, *args, **kwargs):
            
        #Well lenght
        L=5
        n = len(x)
        ExpE = 0; j=0 #Variables to store Expected energy of current solution and norm of current solution
        #Ortogonal term defs
        beta = 100 #10
        orto=0
        #cons=1*10**(-6)
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf
             
        for i in range(n):
            if i==0:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
                j = j + x[i]**2* (L/(n-1))
                orto = orto + x[i]*self.Psi[i]*(L/(n-1))

            elif i==n-1:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
                j = j + x[i]**2 * (L/(n-1))
                orto = orto + x[i]*self.Psi[i]*(L/(n-1))
                
            else:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
                j=j+x[i]**2* (L/(n-1))
                orto = orto + x[i]*self.Psi[i]*(L/(n-1))
                
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        # #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - self.E)**2 + beta* orto**2 + pen


        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]


######### 2nd EXCITATED STATE #################################
class pymooWell2(ElementwiseProblem):
    
    def __init__(self, n_var, E0, Psi0, E1, Psi1, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.E0= E0
        self.Psi0 = Psi0
        self.E1 = E1
        self.Psi1 = Psi1

    def _evaluate(self, x, out, *args, **kwargs):
        E = self.E0 + self.E1 #where E0 E1 is not the actual energy but the differences between consecutive energies En = E{n} - E_{n-1}

        #Well lenght
        L=5
        n = len(x)
        ExpE = 0; j=0
        #Ortogonal term defs
        beta = 10
        orto=0; orto2 = 0
        #cons=1*10**(-6)
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf
        
        for i in range(n):
            if i==0:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
                j = j + x[i]**2* (L/(n-1))
                orto = orto + x[i]*self.Psi1[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi0[i]*(L/(n-1))


            elif i==n-1:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
                j = j + x[i]**2 * (L/(n-1))
                orto = orto + x[i]*self.Psi1[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi0[i]*(L/(n-1))
                
            else:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
                j=j+x[i]**2* (L/(n-1))
                orto = orto + x[i]*self.Psi1[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi0[i]*(L/(n-1))

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps  

        Z = (ExpE/j - E)**2 + beta* orto**2 + beta * orto2**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]


####### 3rd EXCITATED STATE #########################
class pymooWell3(ElementwiseProblem):
    
    def __init__(self, n_var, E0, Psi0, E1, Psi1, E2, Psi2, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.E0= E0
        self.Psi0 = Psi0
        self.E1 = E1
        self.Psi1 = Psi1
        self.E2 = E2
        self.Psi2 = Psi2

    
    def _evaluate(self, x, out, *args, **kwargs):
        E = self.E0 + self.E1 + self.E2
        #Well lenght
        L=5
        n = len(x)
        ExpE = 0; j=0
        #Ortogonal term defs
        beta = 10
        orto=0; orto2 = 0; orto3=0
        #cons=1*10**(-6)
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf
        
        for i in range(n):
            if i==0:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i+2]-2*x[i+1]))
                j = j + x[i]**2* (L/(n-1))
                orto3 = orto3 + x[i]*self.Psi2[i]*(L/(n-1))
                orto = orto + x[i]*self.Psi0[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi1[i]*(L/(n-1))


            elif i==n-1:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i]+x[i-2]-2*x[i-1]))
                j = j + x[i]**2 * (L/(n-1))
                orto3 = orto3 + x[i]*self.Psi2[i]*(L/(n-1))
                orto = orto + x[i]*self.Psi0[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi1[i]*(L/(n-1))
                
            else:
                ExpE = ExpE -0.5*((n-1)/L)*(x[i]*(x[i-1]+x[i+1]-2*x[i]))
                j=j+x[i]**2* (L/(n-1))
                orto3 = orto3 + x[i]*self.Psi2[i]*(L/(n-1))
                orto = orto + x[i]*self.Psi0[i]*(L/(n-1))
                orto2 = orto2 + x[i]*self.Psi1[i]*(L/(n-1))

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + beta* orto**2 + beta * orto2**2 + beta* orto3**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]


#######################################################################################################
#####  Q U A N T U M   H A R M O N I C   O S C I L L A T O R    ################################
#######################################################################################################
#########FUNDAMENTAL STATE######################
class pymoooOscGround(ElementwiseProblem):

    def __init__(self, n_var, grid, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.grid = grid

    def _evaluate(self, x, out, *args, **kwargs):
        "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
        #Energy_lower bound
        E = 0
        #grid size x=[-4, 4]
        n = len(x)
        #x=np.linspace(-4, 4, n)
        dx = (self.grid[-1]-self.grid[0])/(n-1)
        #integral sums
        ExpE = 0; j=0
        cons = 10
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf

        for i in range(n):
            if i==0:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i+2]-2*x[i+1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
        
            elif i==n-1:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i-2]-2*x[i-1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
            
            else:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i-1]+x[i+1]-2*x[i])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]

########## 1er  estado excitado  #####################
class pymoooOsc1(ElementwiseProblem):

    def __init__(self, n_var, grid, E0, Psi0, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.grid = grid
        self.E0 = E0
        self.Psi0 = Psi0

    def _evaluate(self, x, out, *args, **kwargs):
        "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
        #Energy_lower bound
        E = self.E0
        #grid size x=[-4, 4]
        n = len(x)
        #x=np.linspace(-4, 4, n)
        dx = (self.grid[-1]-self.grid[0])/(n-1)
        #integral sums
        ExpE = 0; j=0; orto=0
        cons = 10
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf

        for i in range(n):
            if i==0:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i+2]-2*x[i+1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
        
            elif i==n-1:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i-2]-2*x[i-1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
            
            else:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i-1]+x[i+1]-2*x[i])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + 10*orto**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]

########## 2o  estado excitado  #####################
class pymoooOsc2(ElementwiseProblem):

    def __init__(self, n_var, grid, E0, Psi0, E1, Psi1, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.grid = grid
        self.E0 = E0
        self.Psi0 = Psi0
        self.E1 = E1
        self.Psi1 = Psi1

    def _evaluate(self, x, out, *args, **kwargs):
        "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
        #Energy_lower bound
        E = self.E0 + self.E1
        #grid size x=[-4, 4]
        n = len(x)
        #x=np.linspace(-4, 4, n)
        dx = (self.grid[-1]-self.grid[0])/(n-1)
        #integral sums
        ExpE = 0; j=0; orto=0; orto1=0
        cons = 10
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf

        for i in range(n):
            if i==0:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i+2]-2*x[i+1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx
        
            elif i==n-1:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i-2]-2*x[i-1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx
            
            else:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i-1]+x[i+1]-2*x[i])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + 10*orto**2 + 10*orto1**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]

########## 3er  estado excitado  #####################
class pymoooOsc3(ElementwiseProblem):

    def __init__(self, n_var, grid, E0, Psi0, E1, Psi1, E2, Psi2, eps):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=-1*np.ones(n_var),
                         xu= 1*np.ones(n_var))
        self.eps=eps
        self.grid = grid
        self.E0 = E0
        self.Psi0 = Psi0
        self.E1 = E1
        self.Psi1 = Psi1
        self.E2 = E2
        self.Psi2 = Psi2


    def _evaluate(self, x, out, *args, **kwargs):
        "Estamos discretizando el espacio entre los puntos x=-4 a x=4 con el número de puntos que se pasen como input al EDA"
        #Energy_lower bound
        E = self.E0 + self.E1 + self.E2 #Remember this way En is just the difference in energy respect previous state
        #grid size x=[-4, 4]
        n = len(x)
        #x=np.linspace(-4, 4, n)
        dx = (self.grid[-1]-self.grid[0])/(n-1)
        #integral sums
        ExpE = 0; j=0; orto=0; orto1=0; orto2=0
        cons = 10
        # if (x[0]< -0.1 or x[0] > 0.1) or (x[n-1]<-0.1 or x[n-1]>0.1):
        #     Z = np.inf

        for i in range(n):
            if i==0:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i+2]-2*x[i+1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx
                orto2 = orto2 + x[i]*self.Psi2[i]*dx
        
            elif i==n-1:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i]+x[i-2]-2*x[i-1])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx
                orto2 = orto2 + x[i]*self.Psi2[i]*dx
            
            else:
                ExpE = ExpE + x[i]*dx*(-0.5*(x[i-1]+x[i+1]-2*x[i])/dx**2 + 0.5*self.grid[i]**2*x[i])
                j = j + x[i]**2 * dx
                orto = orto + x[i]*self.Psi0[i]*dx
                orto1 = orto1 + x[i]*self.Psi1[i]*dx
                orto2 = orto2 + x[i]*self.Psi2[i]*dx

        
        ## Boundary condition as penalization term
        pen = 0
        if (x[0] < - self.eps or x[0] > self.eps) or (x[-1] < - self.eps or x[-1] > self.eps):
            #pen = 10e3
            D = np.exp(abs(x[0] - 0)) - 1
            D1 = np.exp(abs(x[-1] - 0)) - 1
            pen = D + D1

        #Avoiding trivial solution. Enforcing norm of state to be 1
        D2 = np.exp(abs(j - 1)) - 1
        pen = pen + D2

        if j <= 0 + self.eps:
            j = j + self.eps

        Z = (ExpE/j - E)**2 + 10*orto**2 + 10*orto1**2 + 10*orto2**2 + pen
        
        # g1 = x[0] - self.eps #This must be <= 0
        # g2 = -x[0] + self.eps #This must be <= 0
        # g3 = x[-1] - self.eps #This must be <= 0
        # g4 = -x[-1] + self.eps #This must be <= 0

        out["F"] = Z
        #out["G"] =  [g1, g2, g3, g4]