# 该程序用于剪切人脸，并存储为灰度图
import os
import cv2
import time

def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

# def reSizeImg(sourcePath_Cas,targetPath_Cas, *suffix):
#     try:
#         ImgPaths = getAllPath(sourcePath_Cas, *suffix)
#         count = 1
#
#         for imgpath in ImgPaths:
#             filename = os.path.split(imgpath)[1]
#             img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
#             f = cv2.resize(img, (200, 200))
#
#             cv2.imwrite(targetPath_Cas + os.sep + filename, f)
#             count += 1
#
#     except IOError:
#         print("Error")
#
#     # 当try块没有出现异常的时候执行p
#     else:
#         print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath_Cas)


# 从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def readPicSaveFace(sourcePath, targetPath,  *suffix):
    try:
        ImagePaths = getAllPath(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # haarcascade_frontalface_alt2.xml为库训练好的分类器文件，下载opencv，安装目录中可找到
        path = "haarcascade_frontalface_alt2.xml" # 级联分类器的地址，换成自己的
        face_cascade = cv2.CascadeClassifier(path)
        for imagePath in ImagePaths:
            # 读灰度图，减少计算
            filename = os.path.split(imagePath)[1]
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if type(img) != str:
                faces = face_cascade.detectMultiScale(img)
                # (x, y)代表人脸区域左上角坐标；
                # w代表人脸区域的宽度(width)；
                # h代表人脸区域的高度(height)。
                for (x, y, w, h) in faces:

                     # 设置人脸宽度大于128像素，去除较小的人脸
                     if w >= 128 and h >= 128:
                        # 扩大图片，可根据坐标调整
                        X = int(x)
                        Y = int(y)
                        W = min(int((x + w)), img.shape[1])
                        H = min(int((y + h)), img.shape[0])
                        f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                        f = cv2.resize(f, (200, 200))
                        cv2.imwrite(targetPath + os.sep + filename, f)
                        count += 1

    except IOError:
        print("Error")

    #当try块没有出现异常的时候执行
    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)

if __name__ == '__main__':
    start = time.time()
    sourcePath = r'G:\FaceRcogImg\TrainSourceImg'# 原始训练数据文件地址，换成自己的
    targetPath = r'G:\FaceRcogImg\TrainFaceData'# 处理后的训练图片的文件存储地址
    readPicSaveFace(sourcePath, targetPath, '.jpg', '.JPG', 'png', 'PNG')
    sourcePath = r'G:\FaceRcogImg\TestSourceImg'# 原始测试数据文件地址，换成自己的
    targetPath = r'G:\FaceRcogImg\TestFaceData'# 处理后的测试图片的文件存储地址
    readPicSaveFace(sourcePath, targetPath, '.jpg', '.JPG', 'png', 'PNG')
    end = time.time()
    print('程序运行时间是：{}'.format(end-start))