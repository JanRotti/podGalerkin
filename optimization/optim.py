import numpy as np

class costateOptimization:
    
    # class initialization
    def __init__(self,dt,timeInterval,T,a0,gamma0,gammaMax,designDim):
        # temporal variables
        self.dt = dt
        self.t0 = timeInterval[0]
        self.tmax = timeInterval[1]
        self.timeInterval = timeInterval
        self.reversedInterval = [self.tmax,self.t0]
        self.T = T

        # initial conditions
        self.a0 = a0
        self.gamma0 = gamma0
        self.gamma_max = gammaMax
        self.dim = a0.shape[0]

        # optimization parameters
        self.design_dim = designDim
        self.design_vars = np.ones(designDim)

        # costate problem
        self.phi0 = np.zeros(a0.shape[0])

        # control function -> only time dependent
        def gamma(t, design_vars=self.design_vars, T=self.T, gamma0=self.gamma0):
            # initial value
            tmp = gamma0 * (1 - t / T)
            # loop design variables
            for m in range(len(design_vars)):
                tmp += design_vars[m] * np.sin(m * np.pi * t * 0.5 / T)
            
            return tmp

        # derivative of control function
        def gammaDerivative(t, T=self.T, gamma0=self.gamma0):
            # initial value
            tmp = - gamma0 * (1 / T)
            # loop design variables
            for m in range(len(design_vars)):
                tmp += design_vars[m] * np.cos(m * np.pi * t * 0.5 / T) * m * np.pi * 0.5 / T
            
            return tmp

        # activation based on time -> utilizing preprocessed activation array
        def activation(t, dt=self.dt):
            # if t between computed values -> interpolation
            if (t % dt != 0):
                return 0.5 * (a[:,int(np.ceil(t/dt))] + a[:,int(np.floor(t/dt))])
            else:
                return a[:,int(t/dt)]


        # Optimization loop
        def optimize(self,learningRate=0.01, maxIter=1e6,epsilon=1e-6):
            lastCost = 0
            cost = 1
            # optimization loop
            for counter in range(maxIter):
                # discontinuation criteria
                if (np.abs(cost - lastCost) < epsilon):
                    break

                lastCost = cost 
                # control trajectory for cost function
                controlTraj = self.control(self.dt,self.timeInterval,self.design_vars,self.gamma0)
                # activation trajectory
                a = self.RK45solver(self.galerkinSystem,self.timeInterval,self.a0,self.dt)
                # costate trajectory -> costate dynamic problem
                phi = self.RK45solver(self.costateProblem,self.reversedInterval,self.phi0,self.dt)
                # gradient function 
                gradient = gradientFunction(phi,timeInterval,self.dt)
                # checking for invalid gradient or cost
                if np.isnan(cost) or np.isnan(gradient).any():
                    print("Optimization stopped due to NaN cost. Latest valid design parameters conserved!")
                    break
                else:
                    print("\r Step:\t", format(counter, '05d'), "\t\t Cost:\t", cost)
                # optimization step
                self.design_vars = self.design_vars - learningRate * gradient

        # RK45 Solver -> trajectory of y
        def RK45solver(self,f,interval,y0,dt=0.0005):
            # temporal
            t = interval[0]
            tmax = interval[1]
            Nt = int((tmax-t)/dt)
            # flag for reverse time
            alt = False
            if Nt < 0:
                Nt *= -1
                dt *= -1
                alt = True
            # initialize trajectory 
            y = np.zeros((y0.shape[0],Nt))
            y[:,0] = y0
            # solver loop
            for i in range(Nt-1):
                k1 = dt * f(t,y[:,i])
                k2 = dt * f(t+dt/2,y[:,i]+k1/2)
                k3 = dt * f(t+dt/2,y[:,i]+k2/2)
                k4 = dt * f(t+dt,y[:,i]+k3)
                k = (k1+2*k2+2*k3+k4)/6
                y[:,i+1] = y[:,i] + k
                t = t + dt  
            # output trajectory
            if alt:
                return np.flip(y,1)
            else:
                return y


        # computing control array for cost function    
        def control(self):
            
            # temporal setting
            dt = self.dt
            t0 = self.t0
            T  = self.T
            tmax = self.tmax  
            Nt = int((tmax-t0)/dt)
            
            # initialize
            tmp = np.zeros(Nt)
            
            # temporal loop
            for i in range(Nt):
                tmp[i] = gamma0 * (1 - (t0 + i * dt) / T)
                # loop design variables
                for m in range(len(self.design_vars)):
                    tmp[i] += self.design_vars[m] * np.sin(m * np.pi * (t0 + i * dt) * 0.5 / T) 
            
            return tmp
            
        # costate dynamics
        def costateProblem(self,t,phi,activation=self.activation,gamma=self.gamma):
            phi_dot = np.zeros_like(phi)
            # compute time dependent activation vector
            a_t = activation(t)
            # loop costates
            for k in range(self.dim):
                # compute time and costate dependent A vector
                A = np.zeros(self.dim)
                for i in range(self.dim):
                    tmp = 0
                    for j in range(self.dim):
                        tmp += (Q_[k][i,j] + Q_[k][j,i]) * a_t[j]
                    A[i] = - (nu * L1[k][i] + L2[k][i] + tmp + g[k,i] * gamma(t))
                # compute costate flow
                phi_dot[k] = np.inner(A,phi) - 0.5 * a_t[k]
            return phi_dot

        # galerkin dynamics
        def galerkinSystem(self,t,a,gamma=self.gamma,dgamma=self.gammaDerivative):
            a_dot = np.empty_like(a)
            # compute time dependent contorl and control derivative
            control = gamma(t)
            controlDerivative = gammaDerivative(t)
            # iterate dof
            for k in range(self.dim):
                a_dot[k] = nu * b1[k] + b2[k] + np.inner((nu * L1[k,:]+L2[k,:]),a) + np.matmul(np.matmul(np.expand_dims(a,1).T,Q_[k]),np.expand_dims(a,1)) + control * (nu * d1[k] + d2[k] + np.inner(g[k],a) + control * f[k]) + h[k] * controlDerivative
            # return activation derivative
            return a_dot

        # continuos adjoint gradient function
        def gradientFunction(self,phi,interval):
            gradient = np.zeros(self.design_dim)
            t0 = self.t0
            tmax = self.tmax  
            Nt = int((tmax-t0)/self.dt)
            # loop design variables
            for m in range(self.design_dim):
                # loop time steps
                for n in range(Nt):    
                    tmp_time = 0
                    # loop system variables
                    for k in range(self.dim):
                        # computing vector based A matrix
                        A = np.zeros(self.dim)
                        for i in range(self.dim):
                            tmp = 0
                            for j in range(self.dim):
                                tmp += (Q_[k][i,j] + Q_[k][j,i]) * a[j,n]
                            A[i] = - (nu * L1[k][i] + L2[k][i] + tmp + g[k][i] * gamma(t0 + n *dt))
                        tmp = 0
                        # computing temporary summations
                        for i in range(self.dim):
                            tmp  += h[k] * A[i] * phi[i,n] + g[k][i] * a[i,n]
                        # computing time cummulative steps
                        tmp_time += (tmp + nu * d1[k] + d2[k] + 2 * gamma(t0 + n * dt) * f[k]) * phi[k][n] + h[k] * 0.5 * a[k][n]
                    # cummulated gradient
                    gradient[m] += dt * (tmp_time + np.exp(np.power(1.5 * gamma(t0 + n * dt),2) - np.power(gamma_max,2)) * 2 * gamma(t0 + n * dt)) * np.sin(m * np.pi * n * dt * 0.5 / T)
            return gradient

        # cost function with costate lagrangian side contraint
        def costateCost(self,a,control,phi,max_control=self.gamma_max):
            # cost function
            cost = np.sum(np.sum(np.square(a),0) + np.exp(np.square(control) - np.square(max_control)) * dt)
            # lagrangian side constraints
            tmp = 0
            # time loop
            for n in range(Nt-1):    
                tmp_time = np.zeros(self.dim)
                # loop system variables
                for k in range(self.dim):
                    tmp_time[k] = (a[k,n+1] - a[k,n]) / dt - nu * b1[k] - b2[k] - np.inner(nu*L1[k]+L2[k],a[:,n]) - np.inner(np.inner(a[:,n],Q_[k]),a[:,n]) - control[n] * (nu * d1[k] + d2[k]) - np.power(control[n],2) * f[k] - control[n] * np.inner(g[k],a[:,n]) - h[k] * self.gammaDerivative(t0 + n * dt)
                tmp -= np.inner(phi[:,n],tmp_time)   
            # return cost with costate constraints
            return cost + tmp