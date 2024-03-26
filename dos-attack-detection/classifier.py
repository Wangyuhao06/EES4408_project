import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection  import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# ******************* START: OWN-CODE ******************* #

npy = "models//3m_weights.npy"
knn_3m = "models//knn_3m.pkl"
knn_tcp = "models//knn_tcp.pkl"
# 加载样本数据集
df_agg = pd.read_csv('dataset/three_moment_tcp_cov_udp_peaks.csv')
x = df_agg[['first_moment', 'second_moment', 'third_moment','tcp_cov','udp_peaks']].values  # 输入特征
y = df_agg[['label']].values  # 目标变量
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1235)  # 数据集分割
# xgboost计算权重
model = xgb.XGBClassifier(verbosity=0, max_depth=5, learning_rate=0.1, n_estimators=50, objective='binary:logistic')
# print learning curve
# train_sizes,train_loss, val_loss = learning_curve(
#     model, X_train, y_train, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1,0.25,0.5,0.75,1]  # 在整个过程中的10%取一次，25%取一次，50%取一次，75%取一次，100%取一次
# )
# # print(train_sizes)
# # print(train_loss)
# # print(val_loss)
# train_loss_mean = -np.mean(train_loss, axis=1)
# val_loss_mean = -np.mean(val_loss,axis=1)
# plt.plot(train_sizes, train_loss_mean, 'o-',color='r',label='Training')
# plt.plot(train_sizes,val_loss_mean,'o-',color='g', label='Cross-validation')
# plt.xlabel('Training examples')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

model.fit(X_train[:,0:3], y_train)

# ----------------------- XGBOOST ----------------------- #
# 对xgb测试集进行预测，并计算准确率
y_pred = model.predict(X_test[:,0:3])
accuracy = accuracy_score(y_test, y_pred)
print("xgb Accuracy: %.2f%%" % (accuracy*100.0))

feature_importance = model.feature_importances_
if sum(feature_importance) == 0:
    weights = 0
else:
    weights = feature_importance/sum(feature_importance)

# ----------------------- XGB + KNN ----------------------- #
# 权重得出分数后KNN判别
score = np.dot(X_train[:,0:3], weights).reshape(-1, 1)
clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                           leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# print learning curve
# train_sizes,train_loss, val_loss = learning_curve(
#     clf, score, y_train, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1,0.25,0.5,0.75,1]  # 在整个过程中的10%取一次，25%取一次，50%取一次，75%取一次，100%取一次
# )
# # print(train_sizes)
# # print(train_loss)
# # print(val_loss)
# train_loss_mean = -np.mean(train_loss, axis=1)
# val_loss_mean = -np.mean(val_loss,axis=1)
# plt.plot(train_sizes, train_loss_mean, 'o-',color='r',label='Training')
# plt.plot(train_sizes,val_loss_mean,'o-',color='g', label='Cross-validation')
# plt.xlabel('Training examples')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

clf.fit(score, y_train.flatten())
# 对测试集进行预测，并计算准确率
y_pred = clf.predict(np.dot(X_test[:,0:3], weights).reshape(-1, 1))
accuracy = accuracy_score(y_test, y_pred)
print("xgb+knn Accuracy: %.2f%%" % (accuracy*100.0))

# # save model
# np.save(npy, weights)
# with open(knn_3m, 'wb') as file:
#     pickle.dump(clf, file)
# xgb_filename = "models//xgb.pkl"
# with open(xgb_filename, 'wb') as file:
#     pickle.dump(model, file)

# ----------------------- PURE-KNN ----------------------- #
clf_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                           leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
# print learning curve
# train_sizes,train_loss, val_loss = learning_curve(
#     clf_knn, score, y_train, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1,0.25,0.5,0.75,1]  # 在整个过程中的10%取一次，25%取一次，50%取一次，75%取一次，100%取一次
# )
# # print(train_sizes)
# # print(train_loss)
# # print(val_loss)
# train_loss_mean = -np.mean(train_loss, axis=1)
# val_loss_mean = -np.mean(val_loss,axis=1)
# plt.plot(train_sizes, train_loss_mean, 'o-',color='r',label='Training')
# plt.plot(train_sizes,val_loss_mean,'o-',color='g', label='Cross-validation')
# plt.xlabel('Training examples')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

clf_knn.fit(X_train[:,0:3], y_train.flatten())
# 对测试集进行预测，并计算准确率
y_pred = clf_knn.predict(X_test[:,0:3])
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy: %.2f%%" % (accuracy*100.0))
knn_filename = knn_tcp
# with open(knn_filename, 'wb') as file:
#     pickle.dump(clf_knn, file)

# ******************* END: OWN-CODE ******************* #