import cv2
import torch
import numpy as np
import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor

# 配置路径
CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "TestImage/testimage2.jpeg"
TEXT_PROMPT = "spoon"

# 加载Grounding DINO模型和图片
image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

# 执行预测：得到边界框和文本标签
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=0.35,
    text_threshold=0.25,
    device=DEVICE,
)

# 标注图片
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# 加载SAM模型
sam_checkpoint = 'Grounded-Segment-Anything/SAM/weights/sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# 设置图像
sam_predictor.set_image(image_source)

# 反归一化坐标：转换为[xyxy]格式
H, W, _ = image_source.shape
boxes = torch.tensor(boxes)
# 将cx, cy, w, h转换为xmin, ymin, xmax, ymax
boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])  # 转换为像素坐标

# 转换为SAM可以使用的框
transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)

# 获取掩码
masks, _, _ = sam_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False  # 单掩码输出更适合多目标场景
)

# 保存掩码mask
def save_masks(masks, output_dir="SegmentImage/Masks"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        # 将掩码转换为 [H, W] 格式，值为 0 或 255 的灰度图
        mask_image = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(output_dir, f"mask_{i+1}.png")
        # 保存为 PNG 格式
        cv2.imwrite(mask_path, mask_image)
        print(f"Saved mask {i+1} to {mask_path}")

# 修正颜色叠加逻辑
def overlay_masks_with_labels(image, masks, boxes, phrases):
    image = image.copy()
    for i, mask in enumerate(masks):
        # 随机生成颜色
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        mask_image = mask.cpu().numpy().astype(np.uint8) * 255

        # 使用 OpenCV 创建半透明掩码
        color_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):  # 添加颜色
            color_mask[:, :, c] = mask_image * (color[c] / 255)

        # 合并原图和掩码（透明度 0.5）
        image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

        # 在框上绘制矩形和标签
        box = boxes[i].cpu().numpy().astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
        label = phrases[i] if i < len(phrases) else f"Object {i+1}"
        cv2.putText(
            image,
            label,
            (box[0], box[1] - 10),  # 框的上方显示文本
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color.tolist(),
            2
        )
    return image

# 保存掩码
save_masks(masks)
# 叠加结果
result_image_with_labels = overlay_masks_with_labels(image_source, masks, boxes_xyxy, phrases)

# 转换颜色通道以适配 OpenCV 的保存
# 教训：OpenCV的颜色通道顺序是BGR，按RGB保存的话输出的颜色会出错
result_image_with_labels = np.clip(result_image_with_labels, 0, 255).astype(np.uint8)
result_image_with_labels_bgr = cv2.cvtColor(result_image_with_labels, cv2.COLOR_RGB2BGR)

# 保存最终带标签的结果
output_path = "SegmentImage/annotated_image_with_labels.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, result_image_with_labels_bgr)
print(f"Segmentation result saved as '{output_path}'")