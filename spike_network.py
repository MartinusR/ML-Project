
import numpy as np

from utility import set_all_args


class SpikeNetwork(object):

    # network parameters
    delta_t = 1e-3
    lamb = 50
    T = 0.5
    sigma_V = 1e-3
    sigma_T = 2e-2
    epsilon_Omega = 1e-4
    epsilon_F = 1e-5
    alpha = 0.21
    beta = 1.25
    mu = 0.02
    gamma = 0.8
    omega = -0.5


    def __init__(self, N, I, x,  F=None, Omega=None, **kwargs):
        """
        x: vector of input
        """

        self.N = N
        self.I = I
        self.x = x

        self.F = F
        if F is None: self.init_F()

        self.Omega = Omega
        if Omega is None: self.init_Omega()

        self.T_vect = self.T * np.ones(N)
        set_all_args(self, kwargs)
        # a list of pairs
        self.tau = 1
        self.V = [np.zeros(self.N)]
        self.o = [np.zeros(self.N)]
        self.r = [np.zeros(self.N)]


    def init_F(self):
        self.F = np.random.randn(self.N, self.I)
        self.F *= self.gamma / np.linalg.norm(self.F, axis=1)[:,None]

    def init_Omega(self):
        self.Omega = self.omega * np.eye(self.N)


    def step(self):
        
        c = (self.x[self.tau] + self.x[self.tau-1] 
            + self.lamb * self.delta_t * self.x[self.tau-1])
        
        V = ((1 - self.lamb * self.delta_t)  * self.V[-1]
            + self.delta_t * np.dot(self.F, c)
            + np.dot(self.Omega, self.o[-1])
            + self.sigma_V * np.random.randn(self.N))

        
        r = (1 - self.lamb*self.delta_t) * self.r[-1] + self.o[-1]

        o = np.zeros(self.N)
        n = np.argmax(V - self.T_vect - self.sigma_V*np.random.randn(self.N))
        
        if V[n] > self.T_vect[n]:
            o[n] = 1
            self.F[n] += (
                self.epsilon_F * (self.alpha * self.x[self.tau-1] - self.F[n]))
            self.Omega[:,n] -= (
                self.epsilon_Omega*(self.beta*(V+self.mu*r)+self.Omega[:,n]))
            self.Omega[n,n] -= self.epsilon_Omega * self.mu

        self.V.append(V)
        self.r.append(r)
        self.o.append(o)

        self.tau += 1


    def simulate(self, iter_num=None):
        if iter_num is None:
            iter_num = len(self.x) - 1
        for _ in range(iter_num):
            self.step()

        
    

