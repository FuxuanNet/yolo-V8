import os
import time
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import gc

# 确保中文显示正常
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def clean_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def tile_image(image, tile_size=640, overlap=128):
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
                iou = compute_iou(box, existing_box)
                if iou > 0.7:  # 更严格的IoU阈值
                    keep = False
                    break

            if keep:
                final_boxes.append(box)
                final_scores.append(score)

        return np.array(final_boxes)

    return []


def merge_predictions(predictions, original_shape, tile_size=640, overlap=128):
    """合并切片的预测结果"""
    height, width = original_shape[:2]
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    # 收集所有预测框
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
                    iou = compute_iou(filtered_box, orig_box)
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


def load_ground_truth(original_name, img_width, img_height):
    """加载真实标签"""
    for folder in ["001-500", "501-1100"]:
        json_path = Path(folder) / f"{original_name}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            break
    else:
        return []

    ground_truth_boxes = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
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


def compute_iou(box1, box2):
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


def calculate_precision_recall(predictions, ground_truth, iou_threshold=0.5):
    """计算精确率和召回率"""
    if len(predictions) == 0 and len(ground_truth) == 0:
        return 1.0, 1.0, 1.0  # 完美情况
    elif len(predictions) == 0:
        return 0.0, 0.0, 0.0  # 没有预测
    elif len(ground_truth) == 0:
        return 0.0, 0.0, 0.0  # 没有真实标签

    # 按置信度排序预测框
    sorted_indices = np.argsort([-box[4] for box in predictions])
    predictions = [predictions[i] for i in sorted_indices]

    true_positives = np.zeros(len(predictions))
    false_positives = np.zeros(len(predictions))
    ground_truth_matched = np.zeros(len(ground_truth))

    for i, pred_box in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(ground_truth):
            if ground_truth_matched[j]:
                continue

            iou = compute_iou(pred_box[:4], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives[i] = 1
            ground_truth_matched[best_gt_idx] = 1
        else:
            false_positives[i] = 1

    # 计算累积精确率和召回率
    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    precision = cumulative_true_positives / (
        cumulative_true_positives + cumulative_false_positives + 1e-6
    )
    recall = cumulative_true_positives / len(ground_truth)

    # 计算AP
    ap = 0
    for i in range(1, len(precision)):
        ap += precision[i] * (recall[i] - recall[i - 1])

    # 计算最后的精确率和召回率
    final_precision = precision[-1]
    final_recall = recall[-1]
    f1_score = (
        2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-6)
    )

    return final_precision, final_recall, f1_score, ap


def calculate_map(
    predictions_all, ground_truth_all, iou_thresholds=np.arange(0.5, 1.0, 0.05)
):
    """计算mAP@0.5:0.95"""
    aps = []

    for iou_threshold in iou_thresholds:
        ap_sum = 0
        valid_count = 0

        for pred, gt in zip(predictions_all, ground_truth_all):
            if len(gt) == 0:
                continue

            # 准备预测数据 (x1, y1, x2, y2, score)
            pred_data = []
            for box, score in zip(pred[0], pred[1]):
                pred_data.append([box[0], box[1], box[2], box[3], score])

            _, _, _, ap = calculate_precision_recall(pred_data, gt, iou_threshold)
            ap_sum += ap
            valid_count += 1

        if valid_count > 0:
            aps.append(ap_sum / valid_count)
        else:
            aps.append(0)

    return np.mean(aps), aps


def measure_inference_speed(
    model, test_images, tile_size=640, overlap=128, iterations=10
):
    """测量推理速度(FPS)"""
    total_time = 0
    total_frames = 0

    # 预热模型
    for img_path in test_images[:5]:
        img = cv2.imread(str(img_path))
        tiles, positions = tile_image(img, tile_size, overlap)
        for tile, pos in zip(tiles, positions):
            model.predict(tile, conf=0.4, verbose=False)
    clean_gpu_memory()

    # 正式测量
    for _ in range(iterations):
        start_time = time.time()
        for img_path in test_images:
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            tiles, positions = tile_image(img, tile_size, overlap)
            total_frames += len(tiles)

            # 单线程处理预测
            all_predictions = []
            for tile, pos in zip(tiles, positions):
                pred = model.predict(tile, conf=0.25, verbose=False)
                if len(pred) > 0 and len(pred[0].boxes) > 0:
                    # 复制boxes对象以避免原地修改
                    boxes_copy = pred[0].boxes.clone()

                    # 调整坐标
                    boxes_copy.xyxy[:, [0, 2]] = boxes_copy.xyxy[:, [0, 2]] + pos[0]
                    boxes_copy.xyxy[:, [1, 3]] = boxes_copy.xyxy[:, [1, 3]] + pos[1]

                    # 创建新的预测对象
                    class AdjustedPred:
                        def __init__(self):
                            self.boxes = boxes_copy

                    all_predictions.append(AdjustedPred())

        end_time = time.time()
        total_time += end_time - start_time
        clean_gpu_memory()

    avg_fps = (total_frames * iterations) / total_time
    return avg_fps


def evaluate_model():
    """评估模型性能"""
    # 加载模型
    model_path = "runs/train/exp8/weights/best.pt"
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建结果目录
    os.makedirs("runs/evaluation", exist_ok=True)
    # 创建检测结果保存目录
    os.makedirs("runs/detect/results", exist_ok=True)

    # 准备测试数据
    val_images = list(Path("dataset/images/val").glob("*.png"))
    print(f"找到 {len(val_images)} 张验证集图片")

    # 获取原始验证集图片
    original_val_images = set()
    for img_path in val_images:
        original_name = img_path.stem.split("_tile_")[0]
        original_val_images.add(original_name)
    original_val_images = list(original_val_images)
    print(f"找到 {len(original_val_images)} 张原始验证图片")

    # 评估指标
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_aps = []
    predictions_all = []
    ground_truth_all = []

    # 处理每张图片
    for original_name in tqdm(original_val_images, desc="评估进度"):
        # 清理GPU内存
        clean_gpu_memory()

        # 找到原始图片
        original_img_path = None
        for folder in ["001-500", "501-1100"]:
            test_path = Path(folder) / f"{original_name}.png"
            if test_path.exists():
                original_img_path = test_path
                break

        if original_img_path is None:
            continue

        # 加载图片
        img = cv2.imread(str(original_img_path))
        if img is None:
            continue

        height, width = img.shape[:2]

        # 切片处理
        tiles, positions = tile_image(img, tile_size=640, overlap=128)

        # 预测
        all_predictions = []
        for tile, pos in zip(tiles, positions):
            pred = model.predict(tile, conf=0.25, verbose=False)
            if len(pred) > 0 and len(pred[0].boxes) > 0:
                # 复制boxes对象以避免原地修改
                boxes_copy = pred[0].boxes.clone()

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

        # 加载真实标签
        ground_truth = load_ground_truth(original_name, width, height)

        # 在原图上绘制结果并保存（与main.py保持一致的可视化风格）
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

        # 保存结果图片到指定目录
        save_path = f"runs/detect/results/{original_name}_result.png"
        cv2.imwrite(save_path, result_img)
        print(f"结果已保存到: {save_path}")

        # 计算指标
        if len(merged_boxes) > 0 and len(ground_truth) > 0:
            # 准备预测数据 (x1, y1, x2, y2, score)
            pred_data = []
            for box, score in zip(merged_boxes, merged_scores):
                pred_data.append([box[0], box[1], box[2], box[3], score])

            precision, recall, f1_score, ap = calculate_precision_recall(
                pred_data, ground_truth
            )
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1_score)
            all_aps.append(ap)

            # 保存用于mAP计算的数据
            predictions_all.append((merged_boxes, merged_scores))
            ground_truth_all.append(ground_truth)
        elif len(merged_boxes) == 0 and len(ground_truth) == 0:
            # 完美情况
            all_precisions.append(1.0)
            all_recalls.append(1.0)
            all_f1_scores.append(1.0)
            all_aps.append(1.0)
        elif len(merged_boxes) == 0:
            # 没有预测
            all_precisions.append(0.0)
            all_recalls.append(0.0)
            all_f1_scores.append(0.0)
            all_aps.append(0.0)
        else:
            # 没有真实标签
            all_precisions.append(0.0)
            all_recalls.append(0.0)
            all_f1_scores.append(0.0)
            all_aps.append(0.0)

        # 清理内存
        del img, tiles, positions, all_predictions
        gc.collect()

    # 计算平均指标
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1_score = np.mean(all_f1_scores)
    avg_ap = np.mean(all_aps)

    # 计算mAP@0.5和mAP@0.5:0.95
    if len(predictions_all) > 0 and len(ground_truth_all) > 0:
        # 计算mAP@0.5
        map_05, _ = calculate_map(
            predictions_all, ground_truth_all, iou_thresholds=[0.5]
        )

        # 计算mAP@0.5:0.95
        map_05_095, _ = calculate_map(predictions_all, ground_truth_all)
    else:
        map_05 = 0.0
        map_05_095 = 0.0

    # 测量推理速度
    print("测量推理速度...")
    # 选择一部分图片用于速度测试
    speed_test_images = []
    for original_name in original_val_images[:5]:  # 只取前5张
        for folder in ["001-500", "501-1100"]:
            test_path = Path(folder) / f"{original_name}.png"
            if test_path.exists():
                speed_test_images.append(test_path)
                break

    if speed_test_images:
        fps = measure_inference_speed(model, speed_test_images)
    else:
        fps = 0.0

    # 输出结果
    print("\n模型评估结果:")
    print(f"精确率 (Precision): {avg_precision:.4f}")
    print(f"召回率 (Recall): {avg_recall:.4f}")
    print(f"F1 分数: {avg_f1_score:.4f}")
    print(f"平均精度 (mAP@0.5): {map_05:.4f}")
    print(f"平均精度 (mAP@0.5:0.95): {map_05_095:.4f}")
    print(f"推理速度 (FPS): {fps:.2f}")

    # 保存结果到文件
    results = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1_score,
        "map_05": map_05,
        "map_05_095": map_05_095,
        "fps": fps,
    }

    with open("runs/evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 绘制指标图
    plt.figure(figsize=(12, 8))

    # 绘制精确率、召回率、F1分数分布图
    plt.subplot(2, 2, 1)
    plt.hist(all_precisions, bins=20, alpha=0.5, label="精确率")
    plt.hist(all_recalls, bins=20, alpha=0.5, label="召回率")
    plt.hist(all_f1_scores, bins=20, alpha=0.5, label="F1分数")
    plt.xlabel("值")
    plt.ylabel("频率")
    plt.title("精确率、召回率、F1分数分布")
    plt.legend()

    # 绘制mAP@0.5:0.95
    plt.subplot(2, 2, 2)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    _, aps = calculate_map(predictions_all, ground_truth_all)
    plt.plot(iou_thresholds, aps, "o-")
    plt.xlabel("IoU阈值")
    plt.ylabel("平均精度 (AP)")
    plt.title("mAP@0.5:0.95")
    plt.grid(True)

    # 绘制关键指标条形图
    plt.subplot(2, 1, 2)
    metrics = ["精确率", "召回率", "F1分数", "mAP@0.5", "mAP@0.5:0.95"]
    values = [avg_precision, avg_recall, avg_f1_score, map_05, map_05_095]
    plt.bar(metrics, values)
    plt.ylim(0, 1.0)
    plt.title("关键指标")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig("runs/evaluation/metrics.png")
    print("评估结果已保存到 runs/evaluation 目录")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    evaluate_model()
