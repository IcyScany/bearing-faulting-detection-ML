import numpy as np
import matplotlib.pyplot as plt
from augment import preprocess
from feature import extract_feature
# -1- 载入数据
path = r"../data/0HP"
data_mark = "FE"
len_data = 1024
overlap_rate = 50  # 50%
random_seed = 1
fs = 12000

X, y = preprocess(path,
                  data_mark,
                  fs,
                  len_data/fs,
                  overlap_rate,
                  random_seed)
# -2- 提取特征
FX, Fy = extract_feature(X, y, fs)

# 绘制007平均值 标签：[0, 1, 4, 7]
avg_feature = {
    'mean': {0: 0, 1: 0, 4: 0, 7: 0},
    'rms': {0: 0, 1: 0, 4: 0, 7: 0},
    'std': {0: 0, 1: 0, 4: 0, 7: 0},
    'skewness': {0: 0, 1: 0, 4: 0, 7: 0},
    'pp': {0: 0, 1: 0, 4: 0, 7: 0},
    'kurtosis': {0: 0, 1: 0, 4: 0, 7: 0},
    'maxf': {0: 0, 1: 0, 4: 0, 7: 0},
    'signal_entropy': {0: 0, 1: 0, 4: 0, 7: 0},
    'am_median_pdf': {0: 0, 1: 0, 4: 0, 7: 0},
}
feat_label = ['均值', '均方根', '标准差', '峰峰值', '偏度值', '峰度', '包络谱最大幅值处的频率', '香农信号熵的无偏估计值', '幅值中位数处的概率密度估计']

# 计算统计值
for num in range(len(Fy)):
    if Fy[num] in [0, 1, 4, 7]:
        for item in avg_feature.keys():
            avg_feature[item][Fy[num]] += np.abs(FX[item][num])

# 绘制统计值
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

index = 1
fig = plt.figure()
plt.subplots_adjust(hspace=0.4)
for key in avg_feature.keys():
    data = avg_feature[key].values()
    plt.subplot(3, 3, index)
    plt.bar(range(4), data, color=['darkorange', 'cornflowerblue', 'darkgray', 'gold'])
    plt.xticks(range(4), labels=['正常', '球体故障', '内圈故障', '外圈故障'])
    plt.title(feat_label[index-1])
    index += 1

# fig.legend(labels=['正常', '球体故障', '内圈故障', '外圈故障'])
plt.show()
