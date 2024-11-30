import json
import cv2

def load_config(config_file='config.json'):
    """加载配置文件，获取最大图像尺寸"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def resize_image(image_path, config_file='config.json'):
    """根据配置文件调整图像大小，使用 OpenCV 读取和缩放图像"""
    # 读取配置
    config = load_config(config_file)
    max_image_size = config.get('max_image_size')

    # 确保最大尺寸在配置中设置了
    if max_image_size is None:
        raise ValueError("错误: 'max_image_size' 参数未找到，检查一下config.json")

    # 使用 OpenCV 读取图像
    image = cv2.imread(image_path)

    # 获取图像的原始宽高
    height, width = image.shape[:2]
    max_dim = max(width, height)

    # 如果最大尺寸超过了设定值，则按比例缩放
    if max_dim > max_image_size:
        scale_factor = max_image_size / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return image