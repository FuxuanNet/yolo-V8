import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def tile_image(image, tile_size=1024, overlap=128):
    """将大图切割成重叠的小块

    Args:
        image: 输入图像
        tile_size: 切片大小
        overlap: 重叠区域大小

    Returns:
        tiles: 切片列表
        positions: 每个切片在原图中的位置 (x, y)
    """
    height, width = image.shape[:2]
    tiles = []
    positions = []

    stride = tile_size - overlap

    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            # 确保最后一块能覆盖到边缘
            end_x = min(x + tile_size, width)
            end_y = min(y + tile_size, height)
            start_x = max(0, end_x - tile_size)
            start_y = max(0, end_y - tile_size)

            tile = image[start_y:end_y, start_x:end_x]
            tiles.append(tile)
            positions.append((start_x, start_y))

    return tiles, positions


def convert_json_to_yolo_tiles(
    json_path, img_width, img_height, tile_size=1024, overlap=128
):
    """将JSON格式的标注转换为YOLO格式，并根据图像切片调整标注"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stride = tile_size - overlap

    # 先生成所有切片的位置信息（与tile_image函数完全一致）
    tile_positions = []
    for y in range(0, img_height - overlap, stride):
        for x in range(0, img_width - overlap, stride):
            # 确保最后一块能覆盖到边缘
            end_x = min(x + tile_size, img_width)
            end_y = min(y + tile_size, img_height)
            start_x = max(0, end_x - tile_size)
            start_y = max(0, end_y - tile_size)
            tile_positions.append((start_x, start_y))

    # 为每个切片创建标注字典，使用索引作为键
    tiles_annotations = {}
    for i in range(len(tile_positions)):
        tiles_annotations[f"tile_{i}"] = []

    # 处理每个标注框
    for shape in data["shapes"]:
        if (
            shape["shape_type"] == "rectangle"
            and "points" in shape
            and len(shape["points"]) == 2
        ):
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 确保坐标顺序正确
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            # 检查每个切片是否与该边界框相交
            for i, (tile_start_x, tile_start_y) in enumerate(tile_positions):
                tile_end_x = tile_start_x + tile_size
                tile_end_y = tile_start_y + tile_size

                # 检查边界框是否与切片相交
                if (
                    x_max > tile_start_x
                    and x_min < tile_end_x
                    and y_max > tile_start_y
                    and y_min < tile_end_y
                ):

                    # 将边界框坐标转换到切片坐标系
                    box_x1 = x_min - tile_start_x
                    box_y1 = y_min - tile_start_y
                    box_x2 = x_max - tile_start_x
                    box_y2 = y_max - tile_start_y

                    # 裁剪到切片边界内
                    box_x1 = max(0, min(box_x1, tile_size))
                    box_y1 = max(0, min(box_y1, tile_size))
                    box_x2 = max(0, min(box_x2, tile_size))
                    box_y2 = max(0, min(box_y2, tile_size))

                    # 确保边界框有效
                    if box_x2 > box_x1 and box_y2 > box_y1:
                        # 计算YOLO格式的标注
                        x_center = (box_x1 + box_x2) / (2 * tile_size)
                        y_center = (box_y1 + box_y2) / (2 * tile_size)
                        width = (box_x2 - box_x1) / tile_size
                        height = (box_y2 - box_y1) / tile_size

                        # 确保值在0-1范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        yolo_line = f"0 {x_center} {y_center} {width} {height}"
                        tiles_annotations[f"tile_{i}"].append(yolo_line)

    return tiles_annotations


def load_ground_truth_boxes(json_path):
    """加载真实标签框 (原始格式，用于可视化)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth_boxes = []
    for shape in data["shapes"]:
        if (
            shape["shape_type"] == "rectangle"
            and "points" in shape
            and len(shape["points"]) == 2
        ):
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 确保坐标顺序正确
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            ground_truth_boxes.append([x_min, y_min, x_max, y_max])

    return np.array(ground_truth_boxes)


def yolo_to_xyxy(yolo_annotation, img_width, img_height):
    """将YOLO格式的标注转换为xyxy格式的坐标"""
    parts = yolo_annotation.strip().split()
    if len(parts) != 5:
        return None

    cls, x_center, y_center, width, height = map(float, parts)

    # 转换为像素坐标
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    # 计算边界框
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2

    return [x1, y1, x2, y2]


def merge_tile_predictions_to_original(
    tiles_annotations,
    tile_positions,
    original_width,
    original_height,
    tile_size=1024,
    overlap=128,
):
    """将切片的标注合并回原图坐标系"""
    merged_boxes = []

    for i, pos in enumerate(tile_positions):
        tile_key = f"tile_{i}"

        if tile_key in tiles_annotations:
            for annotation in tiles_annotations[tile_key]:
                # 将YOLO格式转换为相对于切片的xyxy坐标
                box = yolo_to_xyxy(annotation, tile_size, tile_size)
                if box is None:
                    continue

                # 将切片坐标转换为原图坐标
                x1, y1, x2, y2 = box

                # 使用实际的切片位置
                orig_x1 = x1 + pos[0]
                orig_y1 = y1 + pos[1]
                orig_x2 = x2 + pos[0]
                orig_y2 = y2 + pos[1]

                # 确保坐标在原图范围内
                orig_x1 = max(0, min(orig_x1, original_width))
                orig_y1 = max(0, min(orig_y1, original_height))
                orig_x2 = max(0, min(orig_x2, original_width))
                orig_y2 = max(0, min(orig_y2, original_height))

                # 确保边界框有效
                if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                    merged_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])

    return merged_boxes


def apply_nms(boxes, scores=None, nms_threshold=0.4):
    """应用非极大值抑制，更严格的重复框过滤"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if scores is None:
        scores = np.ones(len(boxes))  # 如果没有置信度分数，使用1.0
    else:
        scores = np.array(scores)

    # 多步NMS策略
    # 第1步：移除完全重复的框（IoU > 0.98）
    unique_boxes = []
    unique_scores = []

    for i, box in enumerate(boxes):
        is_duplicate = False
        for j, existing_box in enumerate(unique_boxes):
            iou = calculate_iou(box, existing_box)
            if iou > 0.98:  # 几乎完全重叠认为是重复
                # 保留置信度更高的（这里都是1.0，保留第一个）
                is_duplicate = True
                break
        if not is_duplicate:
            unique_boxes.append(box)
            unique_scores.append(scores[i])

    if len(unique_boxes) == 0:
        return []

    # 第2步：标准NMS处理高重叠框（IoU > nms_threshold）
    boxes = np.array(unique_boxes)
    scores = np.array(unique_scores)

    # 转换为cv2.dnn.NMSBoxes需要的格式 (x, y, width, height)
    xywh_boxes = boxes.copy()
    xywh_boxes[:, 2] = xywh_boxes[:, 2] - xywh_boxes[:, 0]  # width = x2 - x1
    xywh_boxes[:, 3] = xywh_boxes[:, 3] - xywh_boxes[:, 1]  # height = y2 - y1

    indices = cv2.dnn.NMSBoxes(
        xywh_boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=nms_threshold,
    )

    if len(indices) > 0:
        if isinstance(indices, tuple):
            indices = indices[0]
        if hasattr(indices, "flatten"):
            indices = indices.flatten()
        elif isinstance(indices, (list, tuple)):
            indices = np.array(indices).flatten()
        else:
            indices = [indices]
        final_boxes = boxes[indices]

        # 第3步：最后一次精细过滤，处理中等重叠（IoU > 0.6）
        if len(final_boxes) > 1:
            final_unique = []

            for i, box in enumerate(final_boxes):
                keep = True
                for existing_box in final_unique:
                    if calculate_iou(box, existing_box) > 0.6:  # 中等重叠也过滤
                        keep = False
                        break
                if keep:
                    final_unique.append(box)

            return np.array(final_unique) if final_unique else []
        else:
            return final_boxes

    return []


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label_prefix=""):
    """在图像上绘制边界框"""
    img_with_boxes = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

        # 添加标签
        if label_prefix:
            label = f"{label_prefix}_{i}"
            cv2.putText(
                img_with_boxes,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

    return img_with_boxes


def test_coordinate_conversion():
    """测试坐标转换的正确性"""

    # 创建输出目录
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)

    # 创建三个子目录
    original_dir = test_dir / "1_original_annotations"
    tiles_dir = test_dir / "2_tile_annotations"
    merged_dir = test_dir / "3_merged_annotations"

    for dir_path in [original_dir, tiles_dir, merged_dir]:
        dir_path.mkdir(exist_ok=True)

    # 获取测试图片列表 (选择前5张进行测试)
    test_images = []
    folders = ["001-500", "501-1100"]

    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in sorted(folder_path.glob("*.png"))[:3]:  # 每个文件夹取3张
                if file.with_suffix(".json").exists():
                    test_images.append(file)

    print(f"找到 {len(test_images)} 张测试图片")

    # 测试参数
    tile_size = 1024
    overlap = 128

    for img_path in tqdm(test_images, desc="测试坐标转换"):
        print(f"\n处理图片: {img_path.name}")

        # 加载图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        height, width = img.shape[:2]
        json_path = img_path.with_suffix(".json")

        # 1. 原始标注可视化
        print("  -> 处理原始标注...")
        original_boxes = load_ground_truth_boxes(json_path)
        original_img_with_boxes = draw_boxes(
            img, original_boxes, color=(0, 255, 0), label_prefix="orig"
        )

        # 保存原始标注结果
        original_save_path = original_dir / f"{img_path.stem}_original.png"
        cv2.imwrite(str(original_save_path), original_img_with_boxes)

        # 2. 切片处理和标注转换
        print("  -> 处理切片标注...")
        tiles, positions = tile_image(img, tile_size, overlap)
        tiles_annotations = convert_json_to_yolo_tiles(
            json_path, width, height, tile_size, overlap
        )

        print(f"    切片数量: {len(tiles)}")
        print(f"    标注字典键: {list(tiles_annotations.keys())}")

        # 为每个切片创建可视化
        for i, (tile, pos) in enumerate(zip(tiles, positions)):
            tile_key = f"tile_{i}"

            print(
                f"    切片 {i}: 位置{pos}, 键{tile_key}, 标注数量: {len(tiles_annotations.get(tile_key, []))}"
            )

            if tile_key in tiles_annotations and tiles_annotations[tile_key]:
                # 将YOLO格式转换为切片上的坐标
                tile_boxes = []
                for annotation in tiles_annotations[tile_key]:
                    box = yolo_to_xyxy(annotation, tile_size, tile_size)
                    if box is not None:
                        tile_boxes.append(box)

                if tile_boxes:
                    tile_with_boxes = draw_boxes(
                        tile, tile_boxes, color=(255, 0, 0), label_prefix="tile"
                    )
                    tile_save_path = tiles_dir / f"{img_path.stem}_tile_{i}.png"
                    cv2.imwrite(str(tile_save_path), tile_with_boxes)

        # 3. 合并标注到原图坐标系
        print("  -> 处理合并标注...")
        merged_boxes = merge_tile_predictions_to_original(
            tiles_annotations, positions, width, height, tile_size, overlap
        )

        print(f"    合并前框数量: {len(merged_boxes)}")

        # 应用NMS去除重复框
        if merged_boxes:
            merged_boxes = apply_nms(merged_boxes, nms_threshold=0.5)  # 调整阈值

        print(f"    NMS后框数量: {len(merged_boxes)}")

        # 绘制合并后的结果
        merged_img_with_boxes = draw_boxes(
            img, merged_boxes, color=(0, 0, 255), label_prefix="merged"
        )

        # 保存合并标注结果
        merged_save_path = merged_dir / f"{img_path.stem}_merged.png"
        cv2.imwrite(str(merged_save_path), merged_img_with_boxes)

        # 4. 创建对比图
        print("  -> 创建对比图...")

        # 将三种结果放在一起对比
        comparison_img = np.zeros((height, width * 3, 3), dtype=np.uint8)

        # 原始标注 (绿色)
        comparison_img[:, :width] = original_img_with_boxes

        # 合并标注 (蓝色，更容易区分)
        merged_img_blue = draw_boxes(
            img, merged_boxes, color=(255, 0, 0), label_prefix="merged"  # 蓝色
        )
        comparison_img[:, width : width * 2] = merged_img_blue

        # 同时显示原始和合并的结果 (绿色+红色)
        overlay_img = img.copy()
        overlay_img = draw_boxes(
            overlay_img,
            original_boxes,
            color=(0, 255, 0),
            thickness=2,
            label_prefix="orig",
        )
        overlay_img = draw_boxes(
            overlay_img,
            merged_boxes,
            color=(0, 0, 255),
            thickness=2,
            label_prefix="merge",
        )
        comparison_img[:, width * 2 :] = overlay_img

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            comparison_img, "Original (Green)", (50, 50), font, 1, (255, 255, 255), 2
        )
        cv2.putText(
            comparison_img,
            "Merged (Blue)",
            (width + 50, 50),
            font,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison_img,
            "Overlay (Green+Red)",
            (width * 2 + 50, 50),
            font,
            1,
            (255, 255, 255),
            2,
        )

        comparison_save_path = test_dir / f"{img_path.stem}_comparison.png"
        cv2.imwrite(str(comparison_save_path), comparison_img)

        # 5. 统计信息
        print(f"    原始框数量: {len(original_boxes)}")
        print(f"    合并框数量: {len(merged_boxes)}")

        # 计算IoU来评估转换的准确性
        if len(original_boxes) > 0 and len(merged_boxes) > 0:
            max_ious = []
            for orig_box in original_boxes:
                max_iou = 0
                for merged_box in merged_boxes:
                    iou = calculate_iou(orig_box, merged_box)
                    max_iou = max(max_iou, iou)
                max_ious.append(max_iou)

            avg_iou = np.mean(max_ious)
            print(f"    平均最大IoU: {avg_iou:.3f}")

            # 如果平均IoU小于0.8，说明可能有问题
            if avg_iou < 0.8:
                print(f"    ⚠️  警告: 平均IoU较低，可能存在坐标转换问题!")

    print(f"\n测试完成! 结果保存在 {test_dir} 目录中:")
    print(f"  - {original_dir}: 原始标注可视化")
    print(f"  - {tiles_dir}: 切片标注可视化")
    print(f"  - {merged_dir}: 合并标注可视化")
    print(f"  - 对比图: *_comparison.png")


def calculate_iou(box1, box2):
    """计算两个边界框的交并比(IoU)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou


if __name__ == "__main__":
    print("开始测试坐标转换的正确性...")
    test_coordinate_conversion()
    print("测试完成!")
