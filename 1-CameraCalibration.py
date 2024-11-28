import numpy as np
import cv2
import glob
import json

# 准备标定板角点的对象点，如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# 存储所有图像的对象点和角点
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# 读取目录下所有相机校准棋盘图
images = glob.glob('docs/calibration_images/*.JPG')
print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    # 如果找到足够的角点，则存储对象点和图像点
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 显示内参和畸变系数
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)

# 将相机矩阵和畸变系数转换为列表
camera_params = {
    'camera_matrix': mtx.tolist(),
    'distortion_coefficients': dist.tolist()
}

# 保存为 JSON 文件
with open('camera_caliparams.json', 'w') as json_file:
    json.dump(camera_params, json_file)

# 读取的方式
# with open('camera_params.json', 'r') as json_file:
#     camera_params = json.load(json_file)
# mtx = np.array(camera_params['camera_matrix'])
# dist = np.array(camera_params['distortion_coefficients'])