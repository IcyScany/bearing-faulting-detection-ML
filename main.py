# 变分模态分解
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import matplotlib
import scipy.io
from scipy import fftpack

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

data = scipy.io.loadmat(r'./data/0HP/12k_Drive_End_B007_0_118.mat')
data_list = data['X118_DE_time'].reshape(-1)
signal = data_list[:1024]

# ------- 变分模态分解
t = np.linspace(0, 1, 1024)
T = len(signal)
fs = 1 / T
t = np.arange(1, T + 1) / T

# alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
# 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
alpha = 2000

# 噪声容限，一般取 0, 即允许重构后的信号与原始信号有差别。
tau = 0
# 模态数量  分解模态（IMF）个数
K = 5

# DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1
# DC 若为0则让第一个IMF为直流分量/趋势向量
DC = 0
# 初始化ω值，当初始化为 1 时，均匀分布产生的随机数
# init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
init = 1
# 控制误差大小常量，决定精度与迭代次数
tol = 1e-7

# Apply VMD
# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
# 得到中心频率的数值
print(omega[-1])
# Plot the original signal and decomposed modes
plt.figure(figsize=(15, 10))
plt.subplot(K + 1, 1, 1)
plt.subplots_adjust(hspace=0.5)
plt.plot(t, signal, 'r')
plt.title("原始信号")
for num in range(K):
    plt.subplot(K + 1, 1, num + 2)
    plt.plot(t, u[num, :])
    plt.title("IMF " + str(num + 1))

plt.show()

# --------- ht变换
x = u[3, :]
hx = fftpack.hilbert(x)
plt.plot(x, label=u"Carrier")
plt.plot(np.sqrt(x**2 + hx**2), "r", linewidth=2, label=u"Envelop")
plt.plot(-np.sqrt(x**2 + hx**2), "g", linewidth=2, label=u"-Envelop")
plt.title(u"Hilbert Transform")
plt.legend()
plt.show()

# --------- fft变换
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
