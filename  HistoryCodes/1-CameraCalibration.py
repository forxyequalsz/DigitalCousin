import numpy as np
import cv2
import glob
import json
from libs.resize_image import resize_image

# 调整图像的对比度和亮度
def adjust_contrast_brightness(image, alpha=1.2, beta=15):
    """
    调整图像的对比度和亮度
    :param image: 输入的图像
    :param alpha: 对比度系数 (>1 增强对比度)
    :param beta: 亮度增量 (值越高，图像越亮)
    :return: 增强后的图像
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# 对图像进行锐化
def sharpen_image(image):
    """
    对图像进行锐化
    :param image: 输入的图像
    :return: 锐化后的图像
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# 准备标定板角点的对象点，如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# 存储所有图像的对象点和角点
objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane

# 读取目录下所有相机校准棋盘图
images = glob.glob('docs/calibration_images/*.JPG')
print(f"Found {len(images)} images for calibration.")

for fname in images:
    # 1. 使用 resize_image 函数进行图像缩放
    image = resize_image(fname)  # 调用 resize_image 来调整图像尺寸

    # 检查图像是否为空
    if image is None:
        print(f"Error: Could not load image {fname}. Skipping...")
        continue

    # 2. 对图像进行对比度和亮度调整
    image = adjust_contrast_brightness(image)

    # 3. 对图像进行锐化
    # 效果不佳
    # image = sharpen_image(image)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img', gray)
    cv2.waitKey(500)

    # 寻找棋盘格角点
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), flags)

    # 如果找到足够的角点，则存储对象点和图像点
    if ret:
        # 进一步精细化角点位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"Chessboard corners found and refined for image: {fname}")

        # 绘制并显示角点
        image_with_corners = cv2.drawChessboardCorners(image, (7, 6), corners, ret)
        cv2.imshow('img', image_with_corners)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners NOT found in image: {fname}")

cv2.destroyAllWindows()

# 确保找到了一些角点，objpoints 和 imgpoints 不能为空
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Error: No valid corners were found in any images. Please check the images and try again.")
else:
    # 转换为 np.float32 类型以确保数据格式正确
    objpoints = [np.array(op, dtype=np.float32) for op in objpoints]
    imgpoints = [np.array(ip, dtype=np.float32) for ip in imgpoints]

    # 相机标定
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            # 显示内参和畸变系数
            print("Camera matrix : \n", mtx)
            print("Distortion coefficients : \n", dist)

            # 将相机矩阵和畸变系数转换为列表
            camera_params = {
                'camera_matrix': mtx.tolist(),
                'distortion_coefficients': dist.tolist()
            }

            # 保存为 JSON 文件
            with open('camera_caliparams.json', 'w') as json_file:
                json.dump(camera_params, json_file)
            print("Camera parameters have been saved to camera_caliparams.json")
        else:
            print("Calibration failed. Please check input data.")
    except cv2.error as e:
        print(f"Error occurred during calibration: {e}")