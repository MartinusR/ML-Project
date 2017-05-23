
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from spike_network import SpikeNetwork


def input_signal(I, S, sigma, delta_t, eta):
    Gamma = int(S/delta_t)
    #c = [sigma * np.random.randn(I) for _ in range(Gamma)]
    """
    x = np.zeros(Gamma)
    x[0] = np.zeros(I)
    for _ in range(Gamma-1):
        x = -
    """
    c = []
    for _ in range(I):
        c_tmp = np.random.randn(Gamma)
        #plt.plot(np.linspace(0,S,Gamma),c_tmp)
        #plt.show()
        #fig, ax = plt.subplots()
        window = signal.gaussian(int(eta/delta_t),std=eta/delta_t/7)
        #plt.plot(np.linspace(0,eta,int(eta/delta_t)),window)
        #plt.show()
        c.append(signal.fftconvolve(c_tmp, window, mode='same'))
        #fig, ax = plt.subplots()
        #plt.plot(np.linspace(0,S,Gamma),c[-1])
        #plt.show()
    return np.array(c).reshape(Gamma, I)
    

def simulation(I, S, sigma=2e3, delta_t=1e-3, eta=6e-3):
    x = input_signal(I, S, sigma, delta_t, eta)
    network = SpikeNetwork(20, 2, x, delta_t=delta_t)
    return network
    network.simulate()
