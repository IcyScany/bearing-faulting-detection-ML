#-*- coding:utf-8 -*-
"""
    创建于2020-05-31 14:40:32
    作者：lbb
    描述：对提取的特征进行分类器训练和模型的保存
    其他：相关软件版本
          matplotlib:  3.1.3  
          numpy: 1.18.1  
          scipy: 1.4.1 
          pandas: 1.0.3
          sklearn: 0.22.1 
          joblib: 0.14.1
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Sum, Matern, RationalQuadratic
import joblib         # -> 用来保存模型
from sklearn.metrics import accuracy_score

from utils.augment import preprocess
from utils.feature import extract_feature

# -1- 载入数据
path = r"./data/0HP"
data_mark = "FE"
len_data = 1024
overlap_rate = 50      # -> 50%
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

# -3- 数据集划分
x_train, x_test, y_train, y_test = train_test_split(FX, 
                                                    Fy, 
                                                    test_size=0.10, 
                                                    random_state=2)

# -4- 模型训练和保存
# -4.1- K近邻
knn = make_pipeline(StandardScaler(),
                    KNeighborsClassifier(3))
knn.fit(x_train, y_train)
# 保存Model(models 文件夹要预先建立，否则会报错)
joblib.dump(knn, 'models/knn.pkl')

y_pred = knn.predict(x_test)
score = knn.score(x_test, y_test) * 100
print("knn accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("KNN score is: %.3f" % score, "in test dataset")

# -4.2- SVM
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=0.5))
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
score = svm.score(x_test, y_test) * 100
print("svm accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("svm score is: %.3f" % score,  "in test dataset")
# print(classification_report(y_test, y_pred))

# -4.3- 随机森林
rfc = make_pipeline(StandardScaler(),
                    RandomForestClassifier(max_depth=6, random_state=0))
rfc.fit(x_train, y_train)

joblib.dump(rfc, 'models/RandomForest.pkl')

y_pred = rfc.predict(x_test)
score = rfc.score(x_test, y_test) * 100
print("RandomForest accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("RandomForest score is: %.3f" % score,  "in test dataset")
print(classification_report(y_test, y_pred))

# -4.4- 高斯分布的贝叶斯
nbg = make_pipeline(StandardScaler(), GaussianNB())
nbg.fit(x_train, y_train)
joblib.dump(nbg, 'models/GaussianNB.pkl')

y_pred = nbg.predict(x_test)
score = nbg.score(x_test, y_test) * 100
print("GaussianNB accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("GaussianNB score is: %.3f" % score, "in test dataset")

# -4.5- 高斯 Matérn 内核
nbg = make_pipeline(StandardScaler(),
                    GaussianProcessRegressor(kernel=1.0 * Matern(length_scale=1.0, nu=1.5)))
nbg.fit(x_train, y_train)

joblib.dump(nbg, 'models/Gaussian_Matérn.pkl')

y_pred = nbg.predict(x_test)
y_pred = np.around(y_pred[:]).astype(int)
score = nbg.score(x_test, y_test) * 100
print("Gaussian Matérn accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("Gaussian Matérn score is: %.3f" % score, "in test dataset")

# y_pred = nbg.predict(x_test)
# y_pred = np.around(y_pred[:]).astype(int)
# print(classification_report(y_test, y_pred))

# -4.6- 高斯 有理二次内核
nbg = make_pipeline(StandardScaler(),
                    GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0, alpha=1.5)))
nbg.fit(x_train, y_train)

joblib.dump(nbg, 'models/Gaussian_RationalQuadratic.pkl')

y_pred = nbg.predict(x_test)
y_pred = np.around(y_pred[:]).astype(int)
score = nbg.score(x_test, y_test) * 100
print("Gaussian RationalQuadratic accuracy is: %.5f " % accuracy_score(y_test, y_pred))
print("Gaussian RationalQuadratic score is: %.3f" % score, "in test dataset")
