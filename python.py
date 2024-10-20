
---

### **Complete Code**

Here’s the final, integrated code for the project:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Fourier analysis to extract harmonics
def fourier_analysis(signal, sample_rate):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    xf = xf[:N // 2]
    yf = 2.0 / N * np.abs(yf[:N // 2])
    return xf, yf

# Compute THD
def compute_thd(yf):
    fundamental = yf[0]
    harmonic_sum = np.sqrt(np.sum(yf[1:] ** 2))
    thd = (harmonic_sum / fundamental) * 100
    return thd

# Plot the harmonics
def plot_harmonics(xf, yf, thd):
    plt.figure(figsize=(10, 5))
    plt.bar(xf, yf, width=1.5)
    plt.title(f'Harmonic Analysis (THD: {thd:.2f}%)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Generate sample signal
def generate_signal(fundamental_freq=50, harmonics=[3, 5], sample_rate=1000, duration=1):
    t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    signal = np.sin(2 * np.pi * fundamental_freq * t)
    for harmonic in harmonics:
        signal += 0.3 * np.sin(2 * np.pi * fundamental_freq * harmonic * t)
    return t, signal

# Main function to run the analysis
if __name__ == "__main__":
    t, signal = generate_signal()
    xf, yf = fourier_analysis(signal, 1000)
    thd = compute_thd(yf)
    plot_harmonics(xf, yf, thd)
