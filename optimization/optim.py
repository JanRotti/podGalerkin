import numpy as np

class controlOptimization:

    def __init__(self, designDim, timeInterval, dt, T, a0, phi0, control0, maxControl):
        
        self.designDim = designDim
        self.designVariables = -np.ones(designDim)

        # temporal setup
        self.t0 = timeInterval[0]
        self.tmax = timeInterval[1]
        self.dt = dt
        self.T = T
        self.Nt = int(tmax/dt) + 1

        # initial conditions
        self.a0 = a0
        self.phi0 = phi0
        self.control0 = control0

        # misc
        self.maxControl = maxControl
        self.N = 1 # control weight in cost 
        self.cost = 0

        # optimizaiton control
        self.epsilon = 1e-6
        self.alpha = 1
        self.maxIter = 100000

        # initialize arrays
        self.a = np.zeros((len(a0), Nt))
        self.phi = np.zeros((len(a0), Nt))
        self.gamma = np.zeros(Nt)
        self.grad = np.zeros(len(a0))


    # printing statements
    def status(self,print_string):
        print(f'{print_string:{"-"}<80}')

    # functions for single timestep output
    def control(self,t):
        tmp = self.control0 * (1 - t / self.T)
        for m in range(self.designDim):
            tmp += self.designVariables[m] * np.sin((m+1) * np.pi * t * 0.5 / self.T)
        return tmp

    def controlDerivative(self,t):
        tmp = - self.control0 * (1 / T)
        for m in range(self.designDim):
            tmp += self.designVariables[m] * np.cos((m+1) * np.pi * t * 0.5 / self.T) * m * np.pi * 0.5 / self.T
        return tmp

    def activation(self,t):
        if t % self.dt != 0:
            return 0.5 * (self.a[:,int(np.ceil(t/self.dt))] + self.a[:,int(np.floor(t/self.dt))])
        else:
            return self.a[:,int(t/self.dt)]     

    # RK45 ODE solver for forward/reverse time solve
    def solver(self,f,interval,y0,dt):
        # RK45 Solver -> trajectory of y
        t = interval[0]
        tmax = interval[1]
        Nt = np.abs(int((tmax-t)/dt)) + 1
        # reverse time alternative
        alt = False
        if (tmax < t):
            dt *= -1
            alt = True
        # initial condition
        y = np.zeros((y0.shape[0],Nt))
        y[:,0] = y0
        # stepping loop
        for i in range(Nt-1):
            k1 = dt * f(t,y[:,i])
            k2 = dt * f(t+dt/2,y[:,i]+k1/2)
            k3 = dt * f(t+dt/2,y[:,i]+k2/2)
            k4 = dt * f(t+dt,y[:,i]+k3)
            k = (k1+2*k2+2*k3+k4)/6
            y[:,i+1] = y[:,i] + k
            t = t + dt  
        if alt:
            return np.flip(y,1)
        else:
            return y
    
    # ODE system dynamics
    def costateProblem(self, t, phi):
        n = phi.shape[0]
        phi_dot = np.zeros(n)
        # time step activations
        a = self.activation(t)
        gamma = self.control(t)
        # looping dimensions
        for k in range(n):
            # costate linear vector
            A = np.zeros(n)
            # computing costate vector
            for i in range(n):
                tmp = 0
                for j in range(n):
                    tmp += (Q_[i][k,j] + Q_[i][j,k]) * a[j]
                A[i] = - (nu * L1[i][k] + L2[i][k] + tmp + g[i,k] * gamma)
            phi_dot[k] = np.inner(A,phi) - 0.5 * a[k]
        return phi_dot

    def galerkinSystem(self, t, a):
        n = a.shape[0]
        a_dot = np.zeros(n)
        gamma = self.control(t)
        dgamma = self.controlDerivative(t)
        # iterate dof
        for k in range(n):
            a_dot[k] = nu * b1[k] + b2[k] + np.inner((nu * L1[k,:]+L2[k,:]),a) + np.matmul(np.matmul(np.expand_dims(a,1).T,Q_[k]),np.expand_dims(a,1)) + gamma * (nu * d1[k] + d2[k] + np.inner(g[k],a) + gamma * f[k]) + h[k] * dgamma
        # return activation derivative
        return a_dot

    def gradientFunction(self,phi):
        gradient = np.zeros(self.designDim)
        dim = phi.shape[0]
        # gradient computation loop
        for m in range(self.designDim):
            for n in range(self.Nt):    
                t = self.t0 + n * self.dt
                gamma = self.control(t)
                tmpTime = 0
                for k in range(dim):
                    # computing vector based A
                    A = np.zeros(dim)
                    for i in range(dim):
                        tmp = 0
                        for j in range(dim):
                            tmp += (Q_[i][j,k] + Q_[i][k,j]) * self.a[j,n]
                        A[i] = - (nu * L1[i][k] + L2[i][k] + tmp + g[i][k] * gamma)
                    tmp = 0
                    for i in range(dim):
                        tmp  += h[k] * A[i] * phi[i,n] + g[k][i] * self.a[i,n] * phi[k,n]
                    tmpTime += tmp + (nu * d1[k] + d2[k] + 2 * gamma * f[k]) * phi[k][n] - h[k] * 0.5 * self.a[k][n]
                gradient[m] += self.dt * tmpTime  + self.dt * np.exp(self.N * (np.power(gamma,2) - np.power(gamma_max,2))) * 2 * self.N * gamma * np.sin((m+1) * np.pi * t * 0.5 / self.T)
        return gradient

    def costFunction(self):
        # cost function
        cost = 0
        dim = len(self.a0)
        for n in range(Nt - 1):
            t = self.t0 + n * self.dt
            cost += self.dt * (np.sum(np.square(self.a[:,n])) + np.exp(self.N * (np.power(self.control(t),2)-np.power(self.maxControl,2)))) 
            # lagrangian side constraints
            tmp_time = np.zeros(dim)
            for k in range(dim):
                tmp_time[k] = (self.a[k,n+1] - self.a[k,n]) / self.dt - nu * b1[k] - b2[k] - np.inner(nu*L1[k]+L2[k],self.a[:,n]) - np.inner(np.inner(self.a[:,n],Q_[k]),self.a[:,n]) - self.control(t) * (nu * d1[k] + d2[k]) - np.power(self.control(t),2) * f[k] - self.control(t) * np.inner(g[k],self.a[:,n]) - h[k] * self.controlDerivative(t)
            cost -= self.dt * np.inner(self.phi[:,n],tmp_time)   
        return cost


    def step(self):
        # construct control trajectory
        for n in range(Nt):
            self.gamma[n] = self.control(self.t0 + n * self.dt)
        # solving galerkin system
        self.a = self.solver(self.galerkinSystem,[self.t0,self.tmax],self.a0,self.dt)
        # solving costate ode dynamics in reverse time
        self.phi = self.solver(self.costateProblem,[self.tmax,self.t0],self.phi0,self.dt)
        # computing gradient step function
        self.grad = self.gradientFunction(self.phi)
        # perform update step
        self.designVariables -= self.alpha * self.grad
        # computing cost function
        self.cost = self.costFunction()


    def optimize(self):
        self.status("Optimization started with parameters ")
        self.status("Learning Rate: " + str(self.alpha))
        self.status("Iteration Limit: " + str(self.maxIter))
        # optimization loop
        for s in range(self.maxIter):
            # previouse cost
            prevCost = self.cost
            # computation step
            self.step()
            # step status
            sys.stdout.write("\r"f'{"Iteration: " + str(s):{""}<20}' + f'{"Cost: " + str(self.cost):{""}<60}')
            sys.stdout.flush()
            # discontinuation criterion
            if np.isnan(self.cost):
                self.status("Optimization failed and stopped! ")
                break
            if np.abs(prevCost - self.cost) < self.epsilon:
                self.status("")
                self.status("Optimization converged and finished! ")
                break