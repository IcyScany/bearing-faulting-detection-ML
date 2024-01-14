# 变分模态分解
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import matplotlib
import scipy.io
from scipy import fftpack

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

data = scipy.io.loadmat(r'../data/0HP/12k_Drive_End_IR007_0_105.mat')
data_list = data['X105_DE_time'].reshape(-1)
signal = data_list[:1024]

T = len(signal)
fs = 1 / T
t = np.arange(1, T + 1)

# ------- 变分模态分解 -------
# alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍. 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
alpha = 2000
# 噪声容限，一般取 0, 即允许重构后的信号与原始信号有差别。
tau = 0
# 模态数量  分解模态（IMF）个数
K = 5
# DC 合成信号若无常量，取值为 0；若含常量，则其取值为 1; DC 若为0则让第一个IMF为直流分量/趋势向量
DC = 0
# 初始化ω值，当初始化为 1 时，均匀分布产生的随机数; init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
init = 1
# 控制误差大小常量，决定精度与迭代次数
tol = 1e-7

# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
for num in range(K):
    plt.subplot(K, 1, num + 1)
    plt.plot(u[num], color=[0 / 255, 0 / 255, 255 / 255], linewidth=1)
    plt.title(u"变分模态分解" + "IMF " + str(num + 1))

plt.show()

# --------- ht变换 ---------
ht = np.empty([5, 1024,])
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
for num in range(K):
    x = u[num, :]
    y = fftpack.hilbert(x)
    ht[num] = np.sqrt(x ** 2 + y ** 2)

    plt.subplot(K, 1, num + 1)
    plt.plot(ht[num], color=[0 / 255, 0 / 255, 255 / 255], linewidth=1)
    plt.title(u"包络信号时域图" + "IMF " + str(num + 1))

plt.show()

# --------- fft变换 ---------
fft = np.empty([5, 1024,])
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
for num in range(K):
    N = len(t)  # 采样点数
    y = np.fft.fft(ht[num], N)
    y1 = np.abs(y)  # 取模
    ayy = y1 / (N / 2)  # 换算成实际的幅度
    ayy[0] = ayy[0] / 2
    f = np.arange(0, N) * T / N  # 换算成实际的频率值

    plt.subplot(K, 1, num + 1)
    plt.plot(f[:N // 2], ayy[:N // 2], color=[0 / 255, 0 / 255, 255 / 255], linewidth=1)
    plt.ylabel('幅值(r/min)', fontsize=13)
    plt.title('包络信号频谱图' + "IMF " + str(num + 1))

plt.show()
