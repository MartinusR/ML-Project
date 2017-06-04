import numpy as np
from utility import set_all_args


class SpikeNetwork(object):
    """
    delta_t: time step
    lamb: 1/lamb = charac time, attention to relation between c and lamb
    T: threshold
    alpha: this shall depend on x (big x -> small alpha)
    """

    # network parameters
    delta_t = 1e-3
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

    def __init__(self, N, I, x,  F=None, Omega=None, **kwargs):
        """
        :param N: number of neurons
        :param I: dimension of input
        :param x: vector of input
        :param F: Input weights
        :param Omega: recurrent weights
        :param kwargs: other parameters
        """

        self.N = N
        self.I = I
        self.x = x
        self.c = np.zeros(x.shape)
        self.c[1:, :] = ((self.x[1:] - self.x[:-1]) / self.delta_t
                         + self.lamb * self.x[:-1])
        self.c /= np.std(self.c, axis=0) / 45

        self.F = F
        if F is None:
            self.init_F()

        self.Omega = Omega
        if Omega is None:
            self.init_Omega()

        self.T_vect = self.T * np.ones(N)  # Thresholds
        set_all_args(self, kwargs)
        # a list of pairs
        self.tau = 1
        self.V = [np.zeros(self.N)]         # Membrane potentials through time
        self.o = [np.zeros(self.N)]         # Output spikes
        self.r = [np.zeros(self.N)]             # Filtered output spikes

        self.D = np.zeros((self.I, self.N))     # Decoder

        # Tmp, for debug
        self.spike = None
        self.avg_post = 0
        self.nb_spikes = 0
        self.avg_pre = 0

        self.c = np.zeros((len(x), self.I))
        self.c[:-1] = (((self.x[1:] - self.x[:-1]) / self.delta_t) + 
                     self.lamb * self.x[:-1])
        #self.c = self.c/np.std(self.c[:-1])*20      # ~ 1e2

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

    def compute_another_x(self):
        for i in range(len(self.x)-1):
            self.x[i+1] = (
                self.x[i] + (-self.lamb * self.x[i] 
                + self.c[i]) * self.delta_t)

    def step(self):
        """
        Computes one step of propagation and weight updates.
        """
        # We update the values of inputs and membrane potentials
        #c = ((self.x[self.tau] - self.x[self.tau-1]) / self.delta_t
        #    + self.lamb * self.x[self.tau-1]) #??
        c = self.c[self.tau-1]

        V = ((1 - self.lamb * self.delta_t) * self.V[-1]
            + self.delta_t * np.dot(self.F, c)
            + np.dot(self.Omega, self.o[-1])  # There should be delta_t here, but we don't put it for scaling reasons
            + self.sigma_V * np.random.randn(self.N))

        # There should be the below deltat_t, but for scaling reasons, we don't put it.
        # r = (1 - self.lamb*self.delta_t) * self.r[-1] + self.delta_t * self.o[-1]
        r = (1 - self.lamb*self.delta_t) * self.r[-1] + self.o[-1]

        # TMP: Saves the membrane potentials after each spike
        if self.spike is not None:
            self.avg_post += V[self.spike]
            self.nb_spikes += 1
        self.spike = None

        # We check if a neuron fires a spike
        o = np.zeros(self.N)
        n = np.argmax(V - self.T_vect - self.sigma_T*np.random.randn(self.N))

        if V[n] > self.T_vect[n]:
            # Neuron n fires

            # TMP : saves potential before spike
            self.spike = n
            self.avg_pre += V[n]

            o[n] = 1

            # Updates weights
            self.F[n] += (
                self.epsilon_F*(self.alpha*self.x[self.tau-1]-self.F[n]))
            self.Omega[:, n] -= (
                self.epsilon_Omega*(self.beta*(V+self.mu*r)+self.Omega[:, n]))
            self.Omega[n, n] -= self.epsilon_Omega * self.mu

        self.V.append(V)
        self.r.append(r)
        self.o.append(o)

        self.tau += 1

    def simulate(self, iter_num=None):
        if iter_num is None:
            iter_num = len(self.x) - 1
        for _ in range(iter_num):
            self.step()

    def compute_decoder(self):
        """
        Computes the optimal decoder according to the observed responses and input.
        We use here the explicit formula for optimal D, being given x and r.
        """
        # TODO The decoder may be computed on the signal, but with the final weights ? (And no more updates?)
        avg1 = np.zeros((self.I, self.N))
        avg2 = np.zeros((self.N, self.N))
        nb = len(self.r)

        for i in range(nb):
            r = self.r[i]
            x = self.x[i]

            avg1 += np.outer(x, r)
            avg2 += np.outer(r, r)

        avg1 /= nb
        avg2 /= nb

        self.D = np.dot(avg1, np.linalg.pinv(avg2))

    def decode(self):
        # TODO Idem : May be computed with final weights ?
        return [np.dot(self.D, r) for r in self.r]

    def reset(self, weights=False):
        self.tau = 1
        self.V = [np.zeros(self.N)]
        self.o = [np.zeros(self.N)]
        self.r = [np.zeros(self.N)]

        # Tmp, for debug
        self.spike = None
        self.avg_post = 0
        self.nb_spikes = 0
        self.avg_pre = 0

        if weights:
            self.init_F()
            self.init_Omega()
