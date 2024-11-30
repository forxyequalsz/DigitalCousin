import cv2
import glob
import json
import numpy as np
from libs.resize_image import resize_image

# 设置校准图片路径
image_dir = "docs/calibration_images"
image_paths = glob.glob(f"{image_dir}/*.JPG")

# 设置棋盘格参数
pattern_size = (9, 6)

# 设置亚像素参数（提高精度）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 设置标定点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# 初始化存储矩阵
objpoints = []  # 3D点现实空间
imgpoints = []  # 2D点图像平面

# 检测角点
for image_path in image_paths:
    # 缩放图片，转为灰度图
    image = resize_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        # 亚像素优化
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # 缩放并保存至obj矩阵
        objpoints.append(objp)

        # 可视化检测结果
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        cv2.imshow(f"Detected Corners - {image_path}", image)
        cv2.waitKey(500)

        print(f"Chessboard corners detected in {image_path}")
    else:
        print(f"Chessboard corners NOT detected in {image_path}")

cv2.destroyAllWindows()

# 相机标定
if len(objpoints) > 0 and len(imgpoints) > 0:
    # 获取图片尺寸
    height, width = gray.shape[:2]
    image_size = (width, height)

    # 标定矩阵
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    if ret:
        # 类型转化并存入dict
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "rotation_vectors": [rvec.tolist() for rvec in rvecs],
            "translation_vectors": [tvec.tolist() for tvec in tvecs],
        }

        # 保存json文件
        with open("camera_caliparams.json", "w") as json_file:
            json.dump(calibration_data, json_file, indent=4)

        # 打印内参矩阵检查
        print("Camera calibration successful. Calibration data saved to 'camera_caliparams.json'.")
        print("Camera Matrix:")
        print(camera_matrix)
    else:
        print("Camera calibration failed.")
else:
    print("Insufficient valid chessboard images for calibration.")