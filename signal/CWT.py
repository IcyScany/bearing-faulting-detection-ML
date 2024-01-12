# 连续小波变换处理
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

data3 = scipy.io.loadmat(r'../data/222.mat')
data_list = data3['X222_DE_time'].reshape(-1)
data = data_list[:512]

# 设置连续小波变换参数  设置采样周期为 1/12000，总尺度为 128，小波基函数选择 'cmor1-1'
sampling_period = 1.0 / 12000
total_scal = 128
wave_name = "cmor1-1"

# 计算小波基函数的中心频率 fc
fc = pywt.central_frequency(wave_name)
# 然后根据 total_scal 计算参数 c_param
c_param = 2 * fc * total_scal
# 通过除以 np.arange(total_scal, 0, -1) 来生成一系列尺度值，并存储在 scales 中
scales = c_param / np.arange(total_scal, 0, -1)

# 进行连续小波变换，变换系数存在 coefficients 中， 频率信息存在 frequencies 中
coefficients, frequencies = pywt.cwt(data, scales, wave_name, sampling_period)
# 计算变换系数的幅度 amp
amp = abs(coefficients)
freq_max = frequencies.max()
# 根据采样周期 sampling_period 生成时间轴
t = np.linspace(0, sampling_period, 512, endpoint=False)

plt.contourf(t, frequencies, amp, cmap='jet')
plt.title('轴承-512-128-cmor1-1')
plt.show()

