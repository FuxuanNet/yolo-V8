# YOLOv8 模型保存机制说明

## 1. Best.pt 选择逻辑

### 主要评估指标

YOLOv8 使用以下指标来确定 "best" 模型：

1. **主要指标：mAP@0.5:0.95**
   - 这是在IoU阈值从0.5到0.95（步长0.05）范围内计算的平均精度的平均值
   - 这个指标更全面地评估模型在不同IoU要求下的性能

2. **次要指标（当主要指标相同时）：**
   - mAP@0.5：IoU阈值为0.5时的平均精度
   - Precision：精确率
   - Recall：召回率
   - F1-Score：精确率和召回率的调和平均

### 保存时机

- 每个epoch结束后进行验证
- 如果当前模型的mAP@0.5:0.95指标超过历史最佳，则保存为best.pt
- 同时会显示类似以下的日志：

  ```
  New best mAP@0.5:0.95 = 0.892 (previous best = 0.887)
  Model saved as best.pt
  ```

## 2. 模型保存文件说明

### 默认保存的文件

- **`last.pt`**: 最后一个epoch的模型权重
- **`best.pt`**: 验证集上表现最好的模型权重

### 新增功能：定期保存

通过添加 `save_period=10` 参数，现在还会保存：

- **`epoch10.pt`**: 第10个epoch的模型权重
- **`epoch20.pt`**: 第20个epoch的模型权重
- **`epoch30.pt`**: 第30个epoch的模型权重
- ... 以此类推

## 3. 训练参数详解

```python
results = model.train(
    data="dataset.yaml",
    epochs=100,                # 训练轮数
    imgsz=640,                # 输入图像尺寸
    batch=batch_size,         # 批次大小
    patience=20,              # 早停耐心值（20轮无改善则停止）
    augment=True,             # 启用数据增强
    device=device,            # 使用的设备（GPU/CPU）
    project="runs/train",     # 项目保存目录
    name="exp9",              # 实验名称
    save=True,                # 启用模型保存
    save_period=10,           # 每10轮保存一次模型 ⭐ 新增
    cache=False,              # 不缓存图像
    workers=0,                # 数据加载器工作进程数
    exist_ok=True,            # 允许覆盖现有实验
    pretrained=False,         # 不使用预训练权重
    optimizer="Adam",         # 使用Adam优化器
    close_mosaic=10,          # 最后10轮关闭马赛克增强
    overlap_mask=True,        # 重叠掩码
    mask_ratio=4,             # 掩码比例
    dropout=0.2,              # Dropout率
    label_smoothing=0.1,      # 标签平滑
    cos_lr=True,              # 余弦学习率调度
    warmup_epochs=3,          # 预热轮数
    weight_decay=0.0005,      # 权重衰减
)
```

## 4. 如何使用不同的保存文件

### 使用最佳模型（推荐）

```python
model = YOLO("runs/train/exp9/weights/best.pt")
```

### 使用最新模型

```python
model = YOLO("runs/train/exp9/weights/last.pt")
```

### 使用特定轮次的模型

```python
# 使用第50轮的模型
model = YOLO("runs/train/exp9/weights/epoch50.pt")
```

## 5. 监控训练进度

### 通过日志监控

训练过程中会显示类似以下信息：

```
Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
100/100     0.712G     0.0312     0.0245          0        123        640: 100%|██| 156/156
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all        156        123      0.891      0.887      0.892      0.654
```

### 通过TensorBoard监控

```bash
tensorboard --logdir runs/train/exp9
```

## 6. 模型评估指标说明

- **P (Precision)**: 精确率 = TP/(TP+FP)
- **R (Recall)**: 召回率 = TP/(TP+FN)
- **mAP50**: IoU阈值为0.5时的平均精度
- **mAP50-95**: IoU阈值从0.5到0.95的平均精度
- **box_loss**: 边界框回归损失
- **obj_loss**: 目标检测损失
- **cls_loss**: 分类损失（如果有多类别）

## 7. 最佳实践建议

1. **模型选择**: 优先使用 `best.pt`，它在验证集上表现最好
2. **备份保存**: 定期保存的模型文件可以用于：
   - 分析训练过程中的性能变化
   - 从特定轮次恢复训练
   - 对比不同训练阶段的模型性能
3. **存储管理**: 定期清理不需要的epoch保存文件以节省磁盘空间
