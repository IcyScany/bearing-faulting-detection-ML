# 快速傅里叶变换
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

# 一、读入数据并进行FFT处理
data = scipy.io.loadmat('../data/0HP/12k_Drive_End_OR007@6_0_130.mat')
X130_DE_time = data['X130_DE_time']

fs = 12000  # 采样频率
t = np.arange(0, len(X130_DE_time)/fs, 1/fs)  # 为了去除暂态，取2s到8s的数据

# Plot the original signal in the time domain
plt.figure(1)
plt.plot(t, X130_DE_time, color=[147/255, 0/255, 255/255], linewidth=1)
plt.grid(True)
plt.xlim([0, 10.25])
plt.xlabel('时间（s）')
plt.ylabel('振动加速度(m/s^2)')
plt.title('振动信号时域波形')

# 预处理，去除直流分量，归一化
signal = X130_DE_time.flatten()
x = signal - np.mean(signal)
x = x / np.max(x)

# 做FFT变换
N = len(t)  # 采样点数
y = np.fft.fft(x, N)
y1 = np.abs(y)  # 取模
ayy = y1 / (N / 2)  # 换算成实际的幅度
ayy[0] = ayy[0] / 2
f = np.arange(0, N) * fs / N  # 换算成实际的频率值

# Plot the amplitude-frequency curve
plt.figure(2)
plt.plot(f[:N//2], ayy[:N//2], color=[0/255, 0/255, 255/255], linewidth=1)
plt.grid(True)
plt.xlabel('频率(Hz)', fontsize=13)
plt.ylabel('幅值(r/min)', fontsize=13)
plt.title('幅度-频率曲线图')

plt.show()
