# 变分模态分解
import numpy as np
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
alpha = 2000  # alpha 惩罚系数
tau = 0  # 噪声容限
K = 5  # 模态数量
DC = 0  # DC 合成信号是否含含常量
init = 1  # 初始化ω值
tol = 1e-7  # 控制误差

# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

# --------- ht变换 ---------
ht = np.empty([5, 1024,])
for num in range(K):
    x = u[num, :]
    y = fftpack.hilbert(x)
    ht[num] = np.sqrt(x ** 2 + y ** 2)

# --------- fft变换 ---------
fft = np.empty([5, 1024,])
for num in range(K):
    N = len(t)  # 采样点数
    y = np.fft.fft(ht[num], N)
    y1 = np.abs(y)  # 取模
    ayy = y1 / (N / 2)  # 换算成实际的幅度
    ayy[0] = ayy[0] / 2
    f = np.arange(0, N) * T / N  # 换算成实际的频率值
