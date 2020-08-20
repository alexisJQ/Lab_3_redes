import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy.integrate import quad, simps
from scipy.fftpack import fft, fftfreq, ifft
import os.path as path
import sys
import copy
from math import sqrt

def plot(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

class Audiosignal:
    
    def __init__(self, path):
        self.name = path
        self.sample_rate, self.data = waves.read(path)
        # Number of channels
        if len(np.shape(self.data)) == 1:
            self.number_of_channels = 1
        else:
            self.number_of_channels = np.shape(self.data)[1]
        self.samples = np.shape(self.data)[0]
        self.seconds = float(self.samples / self.sample_rate)
        self.step = float(1 / self.sample_rate)
        self.time_range = np.arange(0, self.seconds, self.step)

    def modulation_AM(self, newsignal = False):
        k = 1
        f = 100
        new_signal = self
        if newsignal:
            new_signal = copy.deepcopy(self)
        new_signal.name = "modulacion AM de {}".format(self.name)
        new_signal.data = k*self.data*np.cos(2*np.pi*f*self.time_range)
        return new_signal

    def modulation_FM(self, newsignal = False):
        k = 1
        f = 100
        new_signal = self
        if newsignal:
            new_signal = copy.deepcopy(self)
        #I = simps(self.data, self.time_range)
        new_signal.name = "modulacion FM de {}".format(self.name)
        new_signal.data = np.cos(2*np.pi*f*self.time_range + k*self.data*self.time_range)
        return new_signal

    def plot_in_time_domain(self, title=None):
        if title is None: 
            title = "señal de {} en el dominio del tiempo".format(self.name)
        plot( 
            self.time_range, 
            np.real(self.data),
            title,
            "tiempo (s)",
            'Amplitud'
        )

    def demodulation_AM(self, newsignal = False):
        k = 1
        f = 100
        new_signal = self
        if newsignal:
            new_signal = copy.deepcopy(self)
        new_signal.name = "demodulacion AM de {}".format(self.name)
        new_signal.data = k*self.data*np.cos(2*np.pi*f*self.time_range)*2
        return new_signal


    def write(self, name=None):
        if name is None:
            name = "{}.wav".format(self.name.replace(" ", "_").replace(".wav", ""))
        data = np.asarray(np.real(self.data), dtype=np.int16)
        waves.write(name, self.sample_rate, data)

if __name__ == "__main__":
    
    path = sys.argv[1]
    original_signal = Audiosignal(path)
    original_signal.plot_in_time_domain()

    # signal modulation
    mod_AM_signal = original_signal.modulation_AM(newsignal=True)
    mod_FM_signal = original_signal.modulation_FM(newsignal=True)
    mod_AM_signal.plot_in_time_domain(title= "modulación AM")
    mod_FM_signal.plot_in_time_domain(title= "modulación FM")


    # AM signal demodulation
    demod_AM_signal = mod_AM_signal.demodulation_AM(newsignal=True)
    demod_AM_signal.plot_in_time_domain(title = "desmodulacion de señal AM")
    mod_AM_signal.write(name=  "AM.wav")
    demod_AM_signal.write(name = "señal demodulada.wav")
    
    plt.show()