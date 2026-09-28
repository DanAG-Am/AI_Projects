import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define variables
amplitude = 1
frequency = 1
phase_shift = 0
sampling_rate = 256
time = 10


def get_sine_wave(freq):
    t = np.linspace(0, time, int(sampling_rate * time), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t + phase_shift)
    return t, sine_wave


def filter_signal(order, low, high, sig):
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def get_psd(sig):
    freq, pxx = signal.welch(
        sig,
        fs=sampling_rate,
        nperseg=sampling_rate,
    )
    db = 10 * np.log10(pxx)
    return freq, db


def plot_amplitude(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.title("Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_psd(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.show()


t_sec, sine = get_sine_wave(10)
t20, sine20 = get_sine_wave(20)
t80, sine80 = get_sine_wave(80)
t120, sine120 = get_sine_wave(120)

sine_total = sine80 + sine20 + sine120

# Plot sine wave
plot_amplitude(t_sec, sine)
plot_amplitude(t120, sine_total)

# Filter signal
clean_signal = filter_signal(
    4,
    1 / sampling_rate,
    80 / sampling_rate,
    sine_total,
)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(t120, sine_total)
ax1.set_title("Signal with noise")
ax1.axis([0, 1, -2, 2])

ax2.plot(t120, clean_signal)
ax2.set_title("After Butterworth 4th-order band-pass filter")
ax2.axis([0, 1, -2, 2])
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

# Get frequency
freq, psd = get_psd(sine_total)

# Plot frequency
plot_psd(freq, psd)