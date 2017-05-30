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
    #plt.ion()
    for _ in range(I):
        c_tmp = sigma * np.random.randn(Gamma)
        #fig, ax = plt.subplots()
        #plt.plot(np.linspace(0,S,Gamma),c_tmp)
        #plt.show()
        window = signal.gaussian(int(eta/delta_t),std=eta/delta_t/7)
        #fig, ax = plt.subplots()
        #plt.plot(np.linspace(0,eta,int(eta/delta_t)),window)
        #plt.show()
        c.append(signal.fftconvolve(c_tmp, window, mode='same'))
        #fig, ax = plt.subplots()
        #plt.plot(np.linspace(0,S,Gamma),c[-1])
        #plt.show()
    return np.transpose(np.array(c))
    

def simulation(I, S, sigma=2e3, delta_t=1e-4, eta=6e-3):
    x = input_signal(I, S, sigma, delta_t, eta)
    network = SpikeNetwork(20, I, x, delta_t=delta_t)
    return x, network


# for the moment being, just call this function with network.o and delta_t
# all of these plot functions will be put in a separate file later
def plot_from_o(o, delta_t):
    evt = events_matrix(np.transpose(o), delta_t * np.arange(len(o)))
    plot_spike_trains(evt, (0, delta_t*len(o)))
    plt.show()


# The result time is in s
def events_matrix(M, T):
    res = []
    for row in M:
        events = []
        for i, s in enumerate(row):
            if s == 1:
                events.append(T[i])
        res.append(events)
    return res


def plot_spike_trains(trains, ts, ax=None, ofs=1.3, lw=0.25, xlab=True):
    if ax is None:
        fig, ax = plt.subplots()
    ax.eventplot(trains, colors=[[0,0,0]], lineoffsets=ofs, lw=lw)
    ax.yaxis.set_visible(False)
    ax.margins(None, 0.01)
    ax.set_xlim(ts[0], ts[1])
    if xlab:
        ax.set_xlabel("time (s)")
    plt.tight_layout()


"""
Probleme :
Si on ne prend pas une signal grand en entree, pas de spike en sortie (pour moins de 1000 !)
Si on prend un signal grand, ca finit par devenir instable avec un neurone qui a un énorme poids Forward, une énorme inhibition
mais qui est insuffisante. Ainsi, par périodes, il fire plein plein de fois.
Mettre mu = 100 règle le probleme, mais bon ça n'est pas du tout satisfaisant !
"""