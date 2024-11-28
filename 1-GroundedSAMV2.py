import cv2
import torch
import numpy as np
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor
from PIL import Image

# 自动选择设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU for computation.")

# 配置路径
CONFIG_PATH = "docs/groundingdino_config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "docs/weight/groundingDINO/groundingdino_swint_ogc.pth"
IMAGE_PATH = "docs/test_images/DSC_3752.JPG"
TEXT_PROMPT = "computer"

# 目标图像缩放的最大宽度和高度
max_width = 1200  # 设置最大宽度
max_height = 1200  # 设置最大高度

# 图像缩放
def resize_image(image_path, max_width, max_height):
    image = Image.open(image_path)
    image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    # 保存缩放后的图像为临时文件
    resized_image_path = "docs/test_images/resized_image.jpg"
    image.save(resized_image_path)
    return resized_image_path

# 执行图像缩放
resized_image_path = resize_image(IMAGE_PATH, max_width, max_height)
print(f"Resized image saved at {resized_image_path}")

# 加载Grounding DINO模型和缩放后图片
image_source, image = load_image(resized_image_path)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
model.to(device)  # 确保模型在正确的设备上

# 执行预测：得到边界框和文本标签
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25,
    device=device,
)

# 使用GroundingDINO的annotate函数绘制边界框和标签
annotated_image = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# 加载SAM模型
sam_checkpoint = 'docs/weight/SAM/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device)  # 确保 SAM 模型在正确的设备上
sam_predictor = SamPredictor(sam)

# 设置图像
sam_predictor.set_image(image_source)

# 反归一化坐标：转换为[xyxy]格式
H, W, _ = image_source.shape
boxes = torch.tensor(boxes).to(device)  # 将边界框数据移动到正确的设备
# 将cx, cy, w, h转换为xmin, ymin, xmax, ymax
boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=device)  # 转换为像素坐标

# 转换为SAM可以使用的框
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

# 获取掩码
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False  # 单掩码输出更适合多目标场景
)

# 保存掩码
def save_masks(masks, output_dir="SegmentImage/Masks"):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        mask_image = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"mask_{i+1}.png")
        cv2.imwrite(mask_path, mask_image)
        print(f"Saved mask {i+1} to {mask_path}")

# 将掩码叠加到已有标注图上
def overlay_masks(image, masks):
    overlay = image.copy()
    for mask in masks:
        mask_image = mask.cpu().numpy().astype(np.uint8) * 255
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # 随机生成颜色
        color_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            color_mask[:, :, c] = mask_image * (color[c] / 255)
        # 将掩码叠加到原图上（透明度 0.5）
        overlay = cv2.addWeighted(overlay, 1, color_mask, 0.5, 0)
    return overlay

# 保存单独的掩码
save_masks(masks)

# 叠加掩码到标注图上（仅叠加掩码，不修改边框与标签）
final_image = overlay_masks(annotated_image, masks)

# 保存最终结果
output_path = "SegmentImage/annotated_image_with_masks.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, final_image)
print(f"Final result saved as '{output_path}'")