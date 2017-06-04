import numpy as np
from utility import set_all_args


class NoMoreDataError(Exception):
    pass


class ExpData(object):

    def __init__(self, N, delta_t, lamb):
        self.tau = 1               # Time
        self.V = [np.zeros(N)]     # Membrane potentials through time
        self.o = [np.zeros(N)]     # Output spikes
        self.r = [np.zeros(N)]     # Filtered output spikes
        self.delta_t = delta_t
        self.lamb = lamb

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x
        self.c = (
            (self.x[1:]-self.x[:-1]) / self.delta_t + self.lamb * self.x[:-1])


class SpikeNetwork(object):
    """
    delta_t: time step
    lamb: 1/lamb = charac time, attention to relation between c and lamb
    T: threshold
    alpha: this shall depend on x (big x -> small alpha)

    Attention:
    delta_t * c should be close to V (so close to T)
    alpha * x should be close to F (and F * x close to V)
    """

    # network parameters
    delta_t = 1e-3
    lamb_x = 50
    lamb = 50
    T = 0.5
    sigma_V = 1e-3
    sigma_T = 2e-2
    epsilon_Omega = 1e-4
    epsilon_F = 1e-5
    alpha = 0.2
    beta = 1.25
    mu = 0.1
    gamma = 0.8
    omega = -0.5

    def __init__(self, N, I, F=None, Omega=None, **kwargs):
        """
        :param N: number of neurons
        :param I: dimension of input
        :param F: Input weights
        :param Omega: recurrent weights
        :param kwargs: other parameters
        """
        self.N = N
        self.I = I
        set_all_args(self, kwargs)

        self.F = F
        if F is None:
            self.init_F()

        self.Omega = Omega
        if Omega is None:
            self.init_Omega()

        self.T_vect = self.T * np.ones(N)  # Thresholds
        self.exps = dict()   


    def init_F(self):
        """
        Initializes input weights randomly, 
        and normalizes them to length gamma.
        """
        self.F = np.random.randn(self.N, self.I)
        self.F *= self.gamma / np.linalg.norm(self.F, axis=1)[:,None]

    def init_Omega(self):
        """
        Initializes recurrent weights to diagonal matrix.
        """
        self.Omega = self.omega * np.eye(self.N)

    def init_exp(self, exp_name):
        self.exps[exp_name] = ExpData(self.N, self.delta_t, self.lamb_x)

    def supply_input(self, exp_name, x, erase=False):
        """
        To simulate the network first supply the input signal
        :param x: vector of input
        """
        if exp_name not in self.exps:
            print("You supply new data")
            self.init_exp(exp_name)
            self.exps[exp_name].x = x
        elif hasattr(self.exps[exp_name], 'x') and not erase:
            print("You add data to an experience that already exists")
            self.exps[exp_name].x = np.append(self.exps[exp_name].x, x, axis=0)
        elif hasattr(self.exps[exp_name], 'x'):
            #print("You erase old data")
            self.init_exp(exp_name)
            self.exps[exp_name].x = x
        else:
            print("You supply new data")
            self.exps[exp_name].x = x

    def get_exp(self, exp_name):
        if exp_name not in self.exps:
            raise ValueError("This experience is not yet set")
        exp = self.exps[exp_name]
        if not hasattr(exp, 'x'):
            raise ValueError("This exprience doesn't have input signal")
        return exp

    def step(self, exp_name, learn=True):
        """
        Computes one step of propagation and weight updates.
        Notice that we don't put self.delta_t with self.o, 
        just a scaling problem
        """
        exp = self.get_exp(exp_name)
        if exp.tau == len(exp.x):
            print("no more input data available, please supply new data or "
                  "reset the experience by running the method init_exp")
            raise NoMoreDataError

        c = exp.c[exp.tau-1]

        # We update the values of inputs and membrane potentials
        V = ((1 - self.lamb * self.delta_t) * exp.V[-1]
            + self.delta_t * np.dot(self.F, c)
            + np.dot(self.Omega, exp.o[-1])
            + self.sigma_V * np.random.randn(self.N))
        r = (1 - self.lamb * self.delta_t) * exp.r[-1] + exp.o[-1]

        # We check if a neuron fires a spike
        o = np.zeros(self.N)
        n = np.argmax(V - self.T_vect - self.sigma_T*np.random.randn(self.N))

        if V[n] > self.T_vect[n]:
            # Neuron n fires
            o[n] = 1
            # Updates weights
            if learn:
                self.update_F(exp, n)
                self.update_Omega(V, r, n)

        exp.V.append(V)
        exp.r.append(r)
        exp.o.append(o)
        exp.tau += 1

    def update_F(self, exp, n):
        self.F[n] += (
            self.epsilon_F * (self.alpha*exp.x[exp.tau-1] - self.F[n]))
        
    def update_Omega(self, V, r, n):
        self.Omega[:, n] -= (
            self.epsilon_Omega * (self.beta*(V+self.mu*r)+self.Omega[:, n]))
        self.Omega[n, n] -= self.epsilon_Omega * self.mu


    def simulate(self, exp_name, learn=True, iter_num=None):
        exp = self.get_exp(exp_name)
        if iter_num is None:
            iter_num = len(exp.x) - exp.tau
        for _ in range(iter_num):
            try:
                self.step(exp_name, learn=learn)
            except NoMoreDataError:
                return

    def respond_signal(self, exp_name, x, learn=False):
        """
        supply the signal, run the simulation, return the result
        mainly when the neuron is already trained to see the result
        """
        self.supply_input(exp_name, x, erase=True)
        self.simulate(exp_name, learn=learn)
        return self.get_exp(exp_name)

    def compute_decoder(self, exp_name):
        """
        Computes the optimal decoder according to the observed responses 
        and input in some experience
        We use here the explicit formula for optimal D, being given x and r.
        """
        exp = self.get_exp(exp_name)
        avg1 = np.zeros((self.I, self.N))
        avg2 = np.zeros((self.N, self.N))
        for i in range(len(exp.x)):
            r = exp.r[i]
            x = exp.x[i]
            avg1 += np.outer(x, r)
            avg2 += np.outer(r, r)
        avg1 /= len(exp.x)
        avg2 /= len(exp.x)
        exp.D = np.dot(avg1, np.linalg.pinv(avg2))

    def decode(self, decoder_name, exp_name):
        exp = self.get_exp(decoder_name)
        if not hasattr(exp, 'D'):
            self.compute_decoder(decoder_name)
        exp2 = self.get_exp(exp_name)
        return exp2.x, [np.dot(exp.D, r) for r in exp2.r]

    def reset(self, weights=False):
        if weights:
            self.init_F()
            self.init_Omega()
        self.exps = dict()
    
    """
    def compute_another_x(self, x):
        for i in range(len(self.x)-1):
            self.x[i+1] = (
                self.x[i] + (-self.lamb * self.x[i] 
                + self.c[i]) * self.delta_t)
    """

