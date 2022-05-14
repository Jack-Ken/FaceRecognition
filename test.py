import numpy as np
import torch
import time
from PIL import Image
from sklearn import preprocessing


# start = time.time()
# IMAGE_SIZE = (200, 200)
# V_img_finall = np.loadtxt(open('./V_img_finall.csv'), delimiter=',', skiprows=0)
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255))  # 默认为范围0~255，拷贝操作
# V_tf = min_max_scaler.fit_transform(np.array(V_img_finall))
#
# end = time.time()
# print('程序运行时间是：{}'.format(end-start))
# for i in range(100):
#     filename = str(i)
#     mean_img1 = np.reshape(V_tf[:, i], IMAGE_SIZE)
#     im = Image.fromarray(np.uint8(mean_img1))
#     # im.show()
#     im.save('{}.jpg'.format(i))


# a = np.array([1,2,3,4])
# a = a.reshape((len(a), 1))
# print(a.shape)
# train_set = np.loadtxt(open('./pca_train_matrix.csv'), delimiter=',', skiprows=0)
# test_set = np.loadtxt(open('./pca_test_matrix.csv'), delimiter=',', skiprows=0)
# test_lable = np.loadtxt(open('./test_label.csv'), delimiter=',', skiprows=0)
# train_lable = np.loadtxt(open('./train_lable.csv'), delimiter=',', skiprows=0)
# number = np.unique(train_lable)
# print(type(number[2]))
# dic = {number[i] : float(i) for i in range(len(number))}
# print(dic)

# train_set = np.transpose(train_set)
# num_label = np.unique(train_lable) # 输出神经元个数
# print(np.array(test_set).shape)
# print(len(train_lable))# 3458
# print(len(test_lable)) # 270