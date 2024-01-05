# 时域图谱
# 参考： https://blog.csdn.net/qq_40949048/article/details/134495407?spm=1001.2014.3001.5502
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False

# 选取"12k Drive End Bearing Fault Data"数据集中故障直径为：0.007、0.014、0.021，负载分别为：0、1、2，外圈故障：@6的驱动端数据
file_names = ['97', '105', '118', '130', '169', '185', '197', '209', '222', '234']
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner',
                'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']

data_12k_1797_10c = pd.DataFrame()
plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.3)

for index in range(10):
    # 十类数据集，每类数据集中只获取DE_time数据
    data = scipy.io.loadmat(f'./data/{file_names[index]}.mat')
    dataList = data[data_columns[index]].reshape(-1)
    data_12k_1797_10c[columns_name[index]] = dataList[:119808]  # 121048  min: 121265

    # 取1024长度数据绘制10分类数据时序图
    plt.subplot(3, 4, index + 1)
    plt.plot(dataList[:1024])
    plt.title(columns_name[index])

print(data_12k_1797_10c.shape)
plt.show()

# # 十分类数据存为csv
# data_12k_1797_10c.set_index('de_normal', inplace=True)
# data_12k_1797_10c.to_csv('./data/data_12k_1797_10c.csv')
