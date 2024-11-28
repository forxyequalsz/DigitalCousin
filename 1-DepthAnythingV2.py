import os
from transformers import pipeline
from PIL import Image
import numpy as np

# 设置 Hugging Face 模型中心的镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载模型和创建管道
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# 打开图像
image = Image.open('docs/test_images/DSC_3752.JPG')

# 如果图像过大，进行缩放（按比例调整大小）
max_width = 1024  # 设置最大宽度
max_height = 1024  # 设置最大高度
image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

# 执行深度估计
depth_result = pipe(image)

# 提取深度图 tensor，转换为 NumPy 数组
depth_tensor = depth_result['predicted_depth']

# 确保 tensor 在 CPU 上，并转换为 numpy 数组
if depth_tensor.is_cuda:
    depth_tensor = depth_tensor.cpu()
depth_array = depth_tensor.numpy()

# 归一化深度值到 0-255 范围，并转换为 uint8
depth_image = Image.fromarray((255 * (depth_array / np.max(depth_array))).astype(np.uint8))

# 保存深度图
depth_image.save("DepthImage/depth_image.png")
print("Depth image saved successfully.")

# 2. 加载相机内参矩阵
import json
with open('camera_caliparams.json', 'r') as f:
    params = json.load(f)
camera_matrix = np.array(params['camera_matrix'])

try:
    # 3. 生成点云
    import open3d as o3d
    # 创建点云
    depth_o3d = o3d.geometry.Image(depth_array.astype(np.uint16))  # Open3D 需要 uint16 类型
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=depth_array.shape[1], height=depth_array.shape[0], 
                                                fx=camera_matrix[0, 0], fy=camera_matrix[1, 1],
                                                cx=camera_matrix[0, 2], cy=camera_matrix[1, 2])
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)

    # 保存点云
    o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
    print("Point cloud saved successfully.")

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Visualization")

except Exception as e:
    print(f"An error occurred: {e}")