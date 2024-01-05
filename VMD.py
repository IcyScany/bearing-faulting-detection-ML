# 变分模态分解
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import matplotlib
import scipy.io

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

data3 = scipy.io.loadmat(r'./data/222.mat')
data_list = data3['X222_DE_time'].reshape(-1)
signal = data_list[:1024]

# -----测试信号及其参数--start-------------
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
# -----测试信号及其参数--end----------

# Apply VMD
# 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
# 得到中心频率的数值
print(omega[-1])
# Plot the original signal and decomposed modes
plt.figure(figsize=(15, 10))
plt.subplot(K + 1, 1, 1)
plt.plot(t, signal, 'r')
plt.title("原始信号")
for num in range(K):
    plt.subplot(K + 1, 1, num + 2)
    plt.plot(t, u[num, :])
    plt.title("IMF " + str(num + 1))

plt.show()
