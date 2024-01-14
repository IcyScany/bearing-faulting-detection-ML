# 希尔伯特变换
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack

t = np.arange(0, 0.3, 1/20000.0)
x = np.cos(2*np.pi*1000*t) * (np.cos(2*np.pi*20*t) + np.cos(2*np.pi*8*t) + 3.0)
hx = fftpack.hilbert(x)
plt.plot(x, label=u"Carrier")
plt.plot(np.sqrt(x**2 + hx**2), "r", linewidth=2, label=u"Envelop")
plt.plot(-np.sqrt(x**2 + hx**2), "g", linewidth=2, label=u"-Envelop")
plt.title(u"Hilbert Transform")
plt.legend()
plt.show()
