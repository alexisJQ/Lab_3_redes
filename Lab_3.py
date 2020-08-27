import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
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
    
    def __init__(self):
        self.name = ""
        self.sample_rate = ""
        self.data = []
        self.samples = 0
        self.seconds = 0
        self.step = 0
        self.time_range = []

    def interpolate(self, samples, seconds=None):
        if samples == self.samples: return
        if seconds is not None: self.seconds = seconds
        i_data_function = interp1d(self.time_range, self.data)
        new_time_range = np.linspace(0, self.seconds, samples)
        self.data = i_data_function(new_time_range)
        self.time_range = new_time_range
        self.samples = samples
        self.sample_rate = self.samples/self.seconds
        self.step = self.time_range[1] - self.time_range[0]


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
    
    def plot_in_freq_domain(self, title=None):
        if title is None:
            title = "Gráfica de la señal {} en el dominio de las frecuencias".format(self.name)
        plot(
            self.freq_domain, 
            np.abs(self.fourier_data),
            title,
            'Frecuencia [Hz]',
            'Amplitud [db]'
        )
    
    def fourier_transform(self):
        self.fourier_data = fft(self.data)
        self.freq_domain = fftfreq(self.samples, d=self.step)

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
        waves.write(name, int(self.sample_rate), data)

def load_audio_from_disk(path):
        signal = Audiosignal()
        signal.name = path
        signal.sample_rate, signal.data = waves.read(path)
        # Number of channels
        if len(np.shape(signal.data)) == 1:
            signal.number_of_channels = 1
        else:
            signal.number_of_channels = np.shape(signal.data)[1]
        signal.samples = np.shape(signal.data)[0]
        signal.seconds = float(signal.samples / signal.sample_rate)
        signal.step = float(1 / signal.sample_rate)
        signal.time_range = np.linspace(0, signal.seconds, signal.samples)
        #self.time_range = np.arange(0, self.seconds, self.step)
        return signal

# Modulates a signal in AM
# signal : Audiosignal to modulate
# k : amplitude factor
# f : carrier's frequency in hertz
# returns an Audiosignal
def am_modulation(signal, k=1, f=10000, fs_factor=2.5):
    sample_rate = int(fs_factor*f)
    if sample_rate < 2*f:
        print("La frecuencia de muestreo debe ser mayor al doble de f ({})".format(f))
        return None

    step = float(1/sample_rate)
    t = np.arange(0, signal.seconds, step)
    carrier_data = k*np.cos(2*np.pi*f*t)
    carrier_samples = len(t)
    signal.interpolate(carrier_samples)
    s = copy.deepcopy(signal)
    s.name = "{} modulada en AM".format(signal.name)
    s.data = carrier_data*signal.data
    return s

# Modulates a signal in FM
# signal : Audiosignal to modulate
# k : frequency variation factor
# f : carrier's frequency in hertz
# returns an Audiosignal
def fm_modulation(signal, k=1, f=10000, fs_factor=2.5):
    sample_rate = int(fs_factor*f)
    if sample_rate < 2*f:
        print("La frecuencia de muestreo debe ser mayor al doble de f ({})".format(f))
        return None
    step = float(1/sample_rate)
    t = np.arange(0, signal.seconds, step)
    samples = len(t)
    signal.interpolate(samples)
    s = copy.deepcopy(signal)
    s.name = "{} modulada en FM".format(signal.name)
    mdt = cumtrapz(s.data, s.time_range, initial=0)
    s.data = np.cos(2*np.pi*f*t + k*mdt)
    return s


if __name__ == "__main__":
    original_signal = load_audio_from_disk("handel.wav")
    original_signal.plot_in_time_domain()

    am = am_modulation(original_signal, k=1, f=10000)
    am.plot_in_time_domain()
    am.fourier_transform()
    am.plot_in_freq_domain()


    fm = fm_modulation(original_signal, k=1, f=10000)
    fm.plot_in_time_domain()
    fm.fourier_transform()
    fm.plot_in_freq_domain()

    demod_AM_signal = am.demodulation_AM(newsignal=True)
    demod_AM_signal.plot_in_time_domain(title = "demodulacion de señal AM")
    
    plt.show()