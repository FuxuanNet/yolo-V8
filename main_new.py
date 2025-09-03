import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import gc
from ultralytics.nn.tasks import DetectionModel


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


def get_image_number(filename):
    """从文件名中提取图片编号"""
    return int(filename.stem[1:])  # 去掉'A'前缀，转换为数字


def clean_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def process_single_image(args):
    """处理单张图片的切片和标注"""
    img_path, is_train, tile_size, overlap = args
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return []

        height, width = img.shape[:2]
        tiles, positions = tile_image(img, tile_size, overlap)

        json_path = str(img_path.with_suffix(".json"))
        if not os.path.exists(json_path):
            print(f"警告：找不到对应的标注文件 {json_path}")
            return []

        tiles_annotations = convert_json_to_yolo_tiles(
            json_path, width, height, tile_size, overlap
        )
        results = []

        for i, (tile, pos) in enumerate(zip(tiles, positions)):
            tile_key = f"tile_{i}"

            split_type = "train" if is_train else "val"
            img_save_path = f"dataset/images/{split_type}/{img_path.stem}_tile_{i}.png"
            label_save_path = (
                f"dataset/labels/{split_type}/{img_path.stem}_tile_{i}.txt"
            )

            annotations = tiles_annotations.get(tile_key, [])
            if annotations:
                # 保存切片图像
                cv2.imwrite(img_save_path, tile)

                # 保存标注文件
                with open(label_save_path, "w") as f:
                    for annotation in annotations:
                        f.write(annotation + "\n")

                results.append((img_save_path, label_save_path, len(annotations)))

        # 清理内存
        del img, tiles, positions, tiles_annotations
        gc.collect()

        return results

    except Exception as e:
        print(f"处理文件 {img_path} 时出错: {e}")
        return []


def prepare_dataset(tile_size=1024, overlap=128):
    """准备数据集，将数据分割为训练集和验证集，并对图像进行切片处理"""
    random.seed(42)
    np.random.seed(42)

    image_files = []
    folders = ["001-500", "501-1100"]

    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in folder_path.glob("*.png"):
                image_files.append(file)

    if not image_files:
        raise Exception("没有找到任何PNG图片文件！请检查数据集路径。")

    image_files = sorted(image_files, key=get_image_number)
    total_images = len(image_files)

    val_size = 220
    step = total_images // val_size

    val_indices = set(range(0, total_images, step))
    if len(val_indices) > val_size:
        val_indices = set(random.sample(list(val_indices), val_size))
    elif len(val_indices) < val_size:
        remaining_indices = set(range(total_images)) - val_indices
        additional_indices = random.sample(
            list(remaining_indices), val_size - len(val_indices)
        )
        val_indices.update(additional_indices)

    train_indices = set(range(total_images)) - val_indices
    train_files = [image_files[i] for i in sorted(train_indices)]
    val_files = [image_files[i] for i in sorted(val_indices)]

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    def process_files(files, is_train):
        """单线程处理文件"""
        for img_path in tqdm(files, desc="处理图片"):
            results = process_single_image((img_path, is_train, tile_size, overlap))
            if results:
                print(f"  -> 成功处理 {len(results)} 个切片")

            # 清理内存
            gc.collect()

    print("\n处理训练集...")
    process_files(train_files, True)

    print("\n处理验证集...")
    process_files(val_files, False)

    print("\n数据集准备完成！")


def create_dataset_yaml():
    """创建数据集配置文件"""
    yaml_content = """
path: dataset  # 数据集根目录
train: images/train  # 训练图片相对路径
val: images/val  # 验证图片相对路径

# 类别数和名称
nc: 1  # 类别数
names: ['droplet']  # 类别名称
    """

    with open("dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print("数据集配置文件创建完成！")


def check_dataset_ready():
    """检查数据集是否已经准备好"""
    required_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
    ]

    # 检查目录是否存在
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            return False

    # 检查文件数量
    train_images = len(list(Path("dataset/images/train").glob("*.png")))
    val_images = len(list(Path("dataset/images/val").glob("*.png")))
    train_labels = len(list(Path("dataset/labels/train").glob("*.txt")))
    val_labels = len(list(Path("dataset/labels/val").glob("*.txt")))

    if train_labels == train_images and val_labels == val_images:
        print(f"\n检测到已存在的数据集:")
        print(f"训练集: {train_images}张图片和标签")
        print(f"验证集: {val_images}张图片和标签")
        return True

    return False


def train_model():
    """训练模型

    模型保存说明：
    1. best.pt: 基于验证集mAP@0.5:0.95指标选择的最佳模型
       - 每个epoch后评估验证集性能
       - 如果当前mAP@0.5:0.95 > 历史最佳值，则保存为best.pt
       - 主要评估指标：mAP@0.5:0.95, mAP@0.5, precision, recall

    2. last.pt: 最后一个epoch的模型权重

    3. epochX.pt: 每10个epoch保存一次的检查点(由save_period=10控制)
       - epoch10.pt, epoch20.pt, epoch30.pt...
    """
    model_path = "yolov8m.pt"
    yaml_path = "E:\\programming\\yolo-V8\\ultralytics\\models\\v8\\yolov8m.yaml"  # 你改过C2f_EMA的yaml文件

    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n使用GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f}GB")

        # 清理GPU内存
        clean_gpu_memory()
    else:
        print("\n警告：未检测到GPU，将使用CPU训练（训练速度会很慢）")

    model = YOLO(yaml_path)  # 初始化结构

    print("\n开始训练...")
    try:
        if device == "0":
            suggested_batch_size = min(12, int(gpu_memory))
            batch_size = max(1, suggested_batch_size)
        else:
            batch_size = 4

        print(f"使用batch_size: {batch_size}")

        results = model.train(
            data="dataset.yaml",
            epochs=100,
            imgsz=640,
            batch=batch_size,
            patience=20,
            augment=True,
            device=device,
            project="runs/train",
            name="exp9",
            save=True,
            # save_period=10,  # YOLOv8不支持此参数，移除
            cache=False,
            workers=0,
            exist_ok=True,
            pretrained=False,
            optimizer="Adam",
            close_mosaic=10,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.2,
            label_smoothing=0.1,
            cos_lr=True,
            warmup_epochs=3,
            weight_decay=0.0005,
        )

        print("\n训练完成！")
        print(f"模型保存目录: runs/train/exp9/weights/")
        print(f"  - best.pt: 验证集mAP@0.5:0.95最佳模型")
        print(f"  - last.pt: 最后一个epoch的模型")
        print(f"  - epochX.pt: 每10个epoch的检查点模型")
        print(f"训练日志和指标图表保存在: runs/train/exp9/")

        # 清理GPU内存
        clean_gpu_memory()

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        raise e


def apply_nms(boxes, scores=None, nms_threshold=0.3):
    """应用非极大值抑制，改进的重复框过滤"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    if scores is None:
        scores = np.ones(len(boxes))  # 如果没有置信度分数，使用1.0
    else:
        scores = np.array(scores)

    # 多步NMS策略
    # 第一步：标准NMS
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
        # 确保indices是numpy数组并且是一维的
        indices = np.array(indices)
        if indices.ndim > 1:
            indices = indices.flatten()
        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]

        # 第二步：额外的重复检测，基于IoU的严格过滤
        final_boxes = []
        final_scores = []

        for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            keep = True
            for existing_box in final_boxes:
                iou = calculate_iou(box, existing_box)
                if iou > 0.7:  # 更严格的IoU阈值
                    keep = False
                    break

            if keep:
                final_boxes.append(box)
                final_scores.append(score)

        return np.array(final_boxes)

    return []


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


def merge_predictions(predictions, original_shape, tile_size=640, overlap=128):
    """合并切片的预测结果"""
    height, width = original_shape[:2]
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    # 使用非极大值抑制合并重叠框
    for pred in predictions:
        if len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()

            merged_boxes.extend(boxes)
            merged_scores.extend(scores)
            merged_classes.extend(classes)

    if merged_boxes:
        merged_boxes = np.array(merged_boxes)
        merged_scores = np.array(merged_scores)
        merged_classes = np.array(merged_classes)

        # 应用改进的NMS
        filtered_boxes = apply_nms(merged_boxes, merged_scores, nms_threshold=0.3)

        if len(filtered_boxes) > 0:
            # 找到对应的分数和类别
            filtered_scores = []
            filtered_classes = []
            for filtered_box in filtered_boxes:
                # 找到最匹配的原始框
                best_match_idx = 0
                best_iou = 0
                for i, orig_box in enumerate(merged_boxes):
                    iou = calculate_iou(filtered_box, orig_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i

                filtered_scores.append(merged_scores[best_match_idx])
                filtered_classes.append(merged_classes[best_match_idx])

            return (
                filtered_boxes,
                np.array(filtered_scores),
                np.array(filtered_classes),
            )

    return [], [], []


def test_model():
    """测试模型并可视化结果"""
    model = YOLO("runs/train/exp9/weights/best.pt")

    os.makedirs("runs/detect/results", exist_ok=True)

    val_images = list(Path("dataset/images/val").glob("*.png"))
    print(f"开始在验证集图片上进行测试...")

    # 获取原始验证集图片
    original_val_images = set()
    for img_path in val_images:
        # 提取原始图片名称（去掉_tile_后缀）
        original_name = img_path.stem.split("_tile_")[0]
        original_val_images.add(original_name)

    for original_name in tqdm(original_val_images):
        # 清理GPU内存
        clean_gpu_memory()

        original_img_path = None
        for folder in ["001-500", "501-1100"]:
            test_path = Path(folder) / f"{original_name}.png"
            if test_path.exists():
                original_img_path = test_path
                break

        if original_img_path is None:
            continue

        img = cv2.imread(str(original_img_path))
        if img is None:
            continue

        height, width = img.shape[:2]
        tiles, positions = tile_image(img, tile_size=640, overlap=128)

        # 单线程处理预测
        all_predictions = []
        for tile, pos in tqdm(
            zip(tiles, positions), desc=f"预测 {original_name}", leave=False
        ):
            pred = model.predict(tile, conf=0.25, verbose=False)
            if len(pred) > 0 and len(pred[0].boxes) > 0:
                # 调整预测框的坐标到原图坐标系
                adjusted_pred = pred[0]

                # 复制boxes对象以避免原地修改
                boxes_copy = adjusted_pred.boxes.clone()

                # 调整坐标
                boxes_copy.xyxy[:, [0, 2]] = boxes_copy.xyxy[:, [0, 2]] + pos[0]
                boxes_copy.xyxy[:, [1, 3]] = boxes_copy.xyxy[:, [1, 3]] + pos[1]

                # 创建新的预测对象
                class AdjustedPred:
                    def __init__(self):
                        self.boxes = boxes_copy

                all_predictions.append(AdjustedPred())

        # 合并预测结果
        merged_boxes, merged_scores, merged_classes = merge_predictions(
            all_predictions, img.shape, tile_size=640, overlap=128
        )

        # 在原图上绘制结果
        result_img = img.copy()
        for box, score, cls in zip(merged_boxes, merged_scores, merged_classes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result_img,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 保存结果
        cv2.imwrite(f"runs/detect/results/{original_name}_result.png", result_img)

        # 清理内存
        del img, tiles, positions, all_predictions
        gc.collect()


def main():
    torch.backends.cudnn.benchmark = True
    """主函数"""
    if not check_dataset_ready():
        print("\n数据集未准备，开始处理...")
        prepare_dataset()
    else:
        print("\n检测到已存在的数据集，跳过处理步骤...")

    if not os.path.exists("dataset.yaml"):
        print("\n创建数据集配置文件...")
        create_dataset_yaml()
    else:
        print("\n检测到已存在的配置文件...")

    print("\n准备开始训练...")
    train_model()

    print("\n开始测试模型...")
    test_model()


if __name__ == "__main__":
    main()
