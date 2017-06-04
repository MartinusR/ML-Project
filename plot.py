import numpy as np
import matplotlib.pyplot as plt

from spike_network import SpikeNetwork
from input_signal import input_signal
from utility import set_all_args


class simulation(object):

    sigma = 10
    delta_t = 1e-3
    eta = 0.1

    def __init__(self, I, S, **kwargs):
        self.I = I
        self.S = S
        set_all_args(self, kwargs)
        self.init_network()
        self.reconstructions_errors = []
        self.iter_num = 0

    def init_network(self):
        self.x = input_signal(
            self.I, self.S, self.sigma, self.delta_t, self.eta)
        self.net = SpikeNetwork(20, self.I, delta_t=self.delta_t, mu=0.02)
        self.net.supply_input('learn', self.x)
        self.exp = self.net.get_exp('learn')

    def run(self, iter_num=None, compute_errs=True):
        if iter_num is None:
            iter_num = len(self.x)-self.iter_num-1
        assert iter_num <= len(self.x)-self.iter_num-1
        for i in range(int(np.ceil(iter_num/1000))):
            nb = min(1000, len(self.x)-i*1000-1)
            self.net.simulate('learn', iter_num=nb)
            if compute_errs:
                exp2 = self.net.respond_signal(
                    'decode', self.x[i*1000:i*1000+nb])
                self.net.compute_decoder('decode')
                x, x_ = self.net.decode('decode', 'decode')
                err = np.linalg.norm(x-x_)
                self.reconstructions_errors.append(err)
            self.iter_num += nb

    def tuning_curves(self, nb_points=100, sample_len=100, radius=10):
        firing_rates = []
        for angle in np.linspace(-180, 180, nb_points):
            x0 = [radius * np.cos(angle / 180 * np.pi), radius * np.sin(angle / 180 * np.pi)]
            x = np.array([x0 for _ in range(sample_len)])
            self.net.init_exp('tuning')
            self.net.supply_input('tuning', x)
            self.net.simulate('tuning', learn=False)
            exp = self.net.get_exp('tuning')
            firing_rates.append(np.sum(exp.o, axis=0))

        return np.array(firing_rates)

    def show_tuning_curves(self, nb_points=100, sample_len=100, radius=10):
        tuning_curves = self.tuning_curves(nb_points=nb_points, sample_len=sample_len, radius=radius)
        plt.plot(np.linspace(-180, 180, nb_points), tuning_curves)
        plt.axis([-180, 180, 0, np.max(tuning_curves)])
        plt.show()

