import cv2
import os
import numpy as np
from PIL import Image
import time

# 处理后的图片像素值是200X200
IMAGE_SIZE = (200, 200)

# 该函数用于获取文件路径矩阵
def getAllPath(dirpath, *suffix):
    PathArray = []
    label = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
                label.append(fn[1:4])

    return PathArray, label

# 该函数用于将所有的灰度图转换为矩阵形式
def LoadData(sourcePath, *suffix):
    ImgPaths, label = getAllPath(sourcePath, *suffix)
    imageMatrix = []
    count = 0
    for imgpath in ImgPaths:
        count += 1
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        # 灰度图矩阵
        mats = np.array(img)
        # 将灰度矩阵转换为向量
        imageMatrix.append(mats.ravel())
    imageMatrix = np.array(imageMatrix)
    return count, imageMatrix, label

def PcaTest(sourcePath, *suffix):
    # count是照片数目，imageMatrix是图片矩阵，label是照片的标签
    count, imageMatrix, label= LoadData(sourcePath, *suffix)
    traib_lable = list(map(int, label))

    # 保存训练样本的标签
    np.savetxt('train_lable.csv', traib_lable, delimiter=',')

    # 数据矩阵，每一列都是一个图像
    imageMatrix = np.transpose(imageMatrix)
    # rows, cols = imageMatrix.shape

    imageMatrix = np.mat(imageMatrix)

    # 原始矩阵的均值行均值
    mean_img = np.mean(imageMatrix, axis=1)

    # 此处用于显示平均脸，如果想要存储到本地，可以自主添加文件存储代码
    mean_img1 = np.reshape(mean_img, IMAGE_SIZE)
    im = Image.fromarray(np.uint8(mean_img1))
    im.show()

    # 均值中心化
    imageMatrix = imageMatrix - mean_img

    # W是特征向量， V是特征向量组 (3458 X 3458)
    imag_mat = (imageMatrix.T * imageMatrix) / float(count)
    W, V = np.linalg.eig(imag_mat)
    # V_img是协方差矩阵的特征向量组
    V_img = imageMatrix * V
    # 降序排序后的索引值
    axis = W.argsort()[::-1]
    V_img = V_img[:, axis]

    number = 0
    x = sum(W)
    for i in range(len(axis)):
        number += W[axis[i]]
        if float(number) / x > 0.9:# 取累加有效值为0.9
            print('累加有效值是：', i) # 前62个特征值保存大部分特征信息
            break
    # 取前62个最大特征值对应的特征向量，组成映射矩阵
    V_img_finall = V_img[:, :62]
    return  V_img_finall, imageMatrix, mean_img, label, count
def recognize(TestsourthPath, V_img_finall, train_imageMatrix, mean_img, train_count, train_lable, *suffix):
    # 读取test矩阵
    test_count, test_imageMatrix, test_label = LoadData(TestsourthPath, *suffix)
    test_label_ = list(map(int, test_label))
    # 存储测试样本标签
    np.savetxt('test_label.csv', test_label_, delimiter=',')
    V_img_finall = np.mat(V_img_finall)

    # 降维后的训练样本空间
    projectedImage = V_img_finall.T * train_imageMatrix
    np.savetxt('pca_train_matrix.csv', projectedImage, delimiter=',')

    # 降维后的测试样本空间
    test_imageMatrix = np.transpose(test_imageMatrix)
    test_imageMatrix = np.mat(test_imageMatrix)
    test_imageMatrix = test_imageMatrix - mean_img
    test_projectedImage = V_img_finall.T * test_imageMatrix
    np.savetxt('pca_test_matrix.csv', test_projectedImage, delimiter=',')

    # 此处通过欧氏距离进行人脸识别，但是准确率较低，不需要的话直接删除
    number = 0
    k = 30
    result = []
    for test in range(test_count):
        distance = []
        for train in range(train_count):
            temp = np.linalg.norm(test_projectedImage[:, test] - projectedImage[:, train])
            distance.append(temp)

        minDistance = min(distance)
        index = distance.index(minDistance)
        result.append(train_lable[index])

        if test_label[test] == train_lable[index]:
            number += 1

    return number/float(test_count), result


if __name__ == '__main__':
    start = time.time()
    sourcePath = r'G:\FaceCasTest'
    TestsourthPath = r'G:\TestFace'
    V_img_finall, imageMatrix, mean_img, train_label, train_count = PcaTest(sourcePath, '.jpg', '.JPG', 'png', 'PNG')
    np.savetxt('V_img_finall.csv', V_img_finall, delimiter=',')
    succsee, result = recognize(TestsourthPath,  V_img_finall, imageMatrix, mean_img, train_count, train_label, '.jpg', '.JPG', 'png', 'PNG')
    end = time.time()
    print('程序运行时间是：{}'.format(end - start))




