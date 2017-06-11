import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

from spike_network import SpikeNetwork
from input_signal import input_signal, plot_from_o
from utility import set_all_args


class simulation(object):
    """
    one should have 
    alpha * x ~ F, F * c * dt ~ 0.1 * T, c ~ lambda * x
    """

    sigma = 10
    delta_t = 1e-3
    eta = 0.1
    N = 20

    def __init__(self, I, S, **kwargs):
        self.I = I
        self.S = S
        set_all_args(self, kwargs)
        self.init_network()
        self.ts = []
        self.reconstruction_errors = []
        self.avg_firing_rates = []
        self.iter_num = 0
        self.ffs = []
        self.cvs = []

    def init_network(self):
        self.x = input_signal(
            self.I, self.S, self.sigma, self.delta_t, self.eta)
        self.net = SpikeNetwork(self.N, self.I, delta_t=self.delta_t, mu=0.02)
        self.net.supply_input('learn', self.x)
        self.exp = self.net.get_exp('learn')

    def adjust_network_parameters(self):
        """
        Please run this except that you're sure that you're using the
        right parameters, the parameters of the network are complicated 
        """
        x_std = np.std(self.x)
        c_std = np.std(self.exp.c)
        self.net.lamb_x = c_std / x_std
        F_order = self.net.T / (self.net.delta_t*c_std*1.2)
        self.net.gamma = F_order
        self.net.init_F()
        self.net.alpha = F_order / (x_std*4)
        self.net.lamb = 8 / self.eta

    def run(self, iter_num=None, step_length=10000,
            compute_errs=True, compute_ffcvs=True):
        if iter_num is None:
            iter_num = len(self.x)-self.iter_num-1
        assert iter_num <= len(self.x)-self.iter_num-1
        for i in range(int(np.ceil(iter_num/step_length))):
            nb = min(step_length, len(self.x)-self.iter_num-1)
            self.net.simulate('learn', iter_num=nb)
            avg_firing_rates = np.mean(np.sum(self.exp.o[-nb:], axis=0))
            self.avg_firing_rates.append(avg_firing_rates)
            if compute_errs:
                exp2 = self.net.respond_signal(
                    'decode', self.x[i*step_length:i*step_length+nb])
                self.net.compute_decoder('decode')
                x, x_ = self.net.decode('decode', 'decode')
                err = np.linalg.norm(x-x_)/step_length
                self.reconstruction_errors.append(err)
            if compute_ffcvs:
                ff, cv = self.compute_stats()
                self.ffs.append(ff)
                self.cvs.append(cv)
            self.ts.append(self.iter_num*self.delta_t)
            self.iter_num += nb
            
    def show_avg_firing_rates(self, smooth=True):
        avg_firing_rates = self.avg_firing_rates
        if smooth:
            avg_firing_rates = wiener(avg_firing_rates, 5)
        plt.plot(self.ts, avg_firing_rates)
        plt.xlabel("time (s)")
        plt.ylabel("firing rate (Hz)")
        plt.show()
        
    def show_reconstruction_errors(self, smooth=True):
        reconstruction_errors = self.reconstruction_errors
        if smooth:
            reconstruction_errors = wiener(reconstruction_errors, 5)
        plt.plot(self.ts, reconstruction_errors)
        plt.xlabel("time (s)")
        plt.ylabel("reconstruction error per second")
        plt.show()
        
    def tuning_curves(self, nb_points=100, sample_len=100, radius=None):
        if radius is None:
            radius = self.sigma
        firing_rates = []
        for angle in np.linspace(-180, 180, nb_points):
            x0 = [radius * np.cos(angle / 180 * np.pi), 
                  radius * np.sin(angle / 180 * np.pi)]
            x = np.array([x0 for _ in range(sample_len)])
            self.net.supply_input('tuning', x, erase=True)
            self.net.simulate('tuning', learn=False)
            exp = self.net.get_exp('tuning')
            firing_rates.append(np.sum(exp.o, axis=0)/sample_len/self.delta_t)

        return np.array(firing_rates)

    def compute_stats(self, nb_points=20, redundancy=10, sample_len=100, 
                      radius=None):
        if radius is None:
            radius = self.sigma
            
        ffs = []
        cvs = []

        for angle in np.linspace(-180, 180, nb_points):
            x0 = [radius * np.cos(angle / 180 * np.pi), 
                  radius * np.sin(angle / 180 * np.pi)]
            x = np.array([x0 for _ in range(sample_len)])
            firing_rates = []
            isi = [[] for _ in range(self.N)]
            for _ in range(redundancy):
                self.net.supply_input('tuning', x, erase=True)
                self.net.simulate('tuning', learn=False)
                exp = self.net.get_exp('tuning')
                firing_rates.append(np.sum(exp.o, axis=0))
                o = np.array(exp.o)

                for neuron in range(len(o[0])):  # self.N
                    firing_times = np.nonzero(o[:, neuron])[0]
                    isi[neuron] += list(np.diff(firing_times))
            ff = np.std(firing_rates, axis=0)**2/np.mean(firing_rates, axis=0)
            cv = np.zeros((self.N,))

            for neuron in range(self.N):
                cv[neuron] = np.std(isi[neuron]) / np.mean(isi[neuron])
            # We Replace nan with zeros
            # ff = np.nan_to_num(ff)
            # cv = np.nan_to_num(cv)

            ffs.append(ff)
            cvs.append(cv)
        ffs = np.array(ffs)
        cvs = np.array(cvs)
        return np.mean(ffs[~np.isnan(ffs)]), np.mean(cvs[~np.isnan(cvs)])

    def show_tuning_curves(self, nb_points=100, sample_len=100, radius=None):
        if radius is None:
            radius = self.sigma
        tuning_curves = self.tuning_curves(
            nb_points=nb_points, sample_len=sample_len, radius=radius)
        plt.plot(np.linspace(-180, 180, nb_points), tuning_curves)
        plt.axis([-180, 180, 0, np.max(tuning_curves)])
        plt.ylim(0,1000)
        plt.xlabel("angle (degree)")
        plt.ylabel("firing rate (Hz)")
        plt.show()
        
    def plot_ff_cv(self):
        plt.plot(self.ts, self.ffs, label="fano factor")
        plt.plot(self.ts, self.cvs, label="coefficient of variation")
        plt.xlabel("time (s)")
        plt.legend()
        plt.show()

    def plot_os(self, start=0, length=1000, y=None):
        if y is None:
            y = self.x[start:start+length]
        spike_exp = self.net.respond_signal('spike_trains', y)
        plot_from_o(spike_exp.o, self.delta_t)

    def plot_Vs(self, start=0, length=1000, y=None):
        if y is None:
            y = self.x[start:start+length]
        V_exp = self.net.respond_signal('potentials', y)
        plt.plot(self.delta_t*np.arange(V_exp.tau), V_exp.V)
        plt.xlabel("time (s)")
        plt.ylabel("membrane potential")
        plt.margins(0, 0.5)
        plt.show()

    def reconstruct(self, start=0, length=1000, y=None):
        if y is None:
            y = self.x[start:start+length]
        rec = self.net.respond_signal('reconstruct', y)
        z, z_ = self.net.decode('reconstruct', 'reconstruct')
        plt.plot(z)
        plt.plot(z_)
        plt.show()

