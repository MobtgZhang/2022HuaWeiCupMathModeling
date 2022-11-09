#sklearn里使用AdaBoostRegressor类对AdaBoost回归算法进行实现

#这里选择叠加正弦曲线，并加上高斯噪声的方式来创建数据集。
#这样的数据集非常适合用来测试、对比和可视化回归算法的性能。代码如下：
import numpy as np
import matplotlib.pyplot as plt
# 创建随机数种子
from sklearn.ensemble import AdaBoostRegressor
#回归决策树
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(111)
# 训练集X为300个0到10之间的随机数
X = np.linspace(0, 10, 300)[:, np.newaxis]
# 定义训练集X的目标变量
y = np.sin(1*X).ravel() + np.sin(2*X).ravel() + np.sin(3* X).ravel()+np.cos(3*X).ravel() +rng.normal(0, 0.3, X.shape[0])

#画出坐标图
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, c='k', label='data', s=10, zorder=1, edgecolors=(0, 0, 0))
# plt.xlabel("X")
# plt.ylabel("y", rotation=0)
# plt.show()

# 固定基学习器最大深度 定义不同迭代次数的AdaBoost回归器模型 最大深度为4，调节迭代次数分别为1、10和100
# adbr_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=1, random_state=123)
# adbr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=10, random_state=123)
# adbr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=123)

# 固定迭代次数为100，调节基学习器（回归决策树）的最大深度，分别为4、5、6，对比拟合效果。
adbr_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=123)
adbr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=100, random_state=123)
adbr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=100, random_state=123)
#实际使用中，在控制拟合时间的前提下，读者应该尽量将基学习器回归决策树的最大深度设置得大一点，然后再在此基础上尝试对n_estimators等参数进行调参。

# 拟合上述三个模型
adbr_1.fit(X, y)
adbr_2.fit(X, y)
adbr_3.fit(X, y)

# 读取各个模型的最大迭代次数
adbr_1_n_estimators = adbr_1.get_params('n_estimators')
adbr_2_n_estimators = adbr_2.get_params('n_estimators')
adbr_3_n_estimators = adbr_3.get_params('n_estimators')


# 预测
y_1 = adbr_1.predict(X)
y_2 = adbr_2.predict(X)
y_3 = adbr_3.predict(X)
# 画出各个模型的回归拟合效果
plt.figure(figsize=(10, 6))
# 画出训练数据集（用黑色表示）
plt.scatter(X, y, c="k", s=10, label="Training Samples")
# 画出adbr_1模型（最大迭代次数为1)的拟合效果（用红色表示）
plt.plot(X, y_1, c="r", label="n_estimators=%d" % adbr_1_n_estimators.get('n_estimators'), linewidth=1)
# 画出adbr_2模型（最大迭代次数为10)的拟合效果（用绿色表示）
plt.plot(X, y_2, c="g", label="n_estimators=%d" % adbr_1_n_estimators.get('n_estimators'), linewidth=1)
# 画出adbr_3模型（最大迭代次数为100)的拟合效果（用蓝色表示）
plt.plot(X, y_3, c="b", label="n_estimators=%d" % adbr_1_n_estimators.get('n_estimators'), linewidth=1)

plt.xlabel("data")
plt.ylabel("target")
plt.title("AdaBoost_Regressor Comparison with different n_estimators when max_depth=3")
plt.legend()
plt.show()


