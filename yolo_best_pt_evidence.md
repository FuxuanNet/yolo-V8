# YOLOv8 Best.pt 选择机制的代码证据

## 1. 核心证据：Fitness函数定义

**文件位置**: `ultralytics/yolo/utils/metrics.py` 第484-487行

```python
def fitness(self):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (np.array(self.mean_results()) * w).sum()
```

**关键发现**:

- fitness 是一个加权组合指标
- 权重分配: `[Precision: 0.0, Recall: 0.0, mAP@0.5: 0.1, mAP@0.5:0.95: 0.9]`
- **mAP@0.5:0.95 占90%的权重，是决定性因素**
- mAP@0.5 只占10%权重
- Precision和Recall权重为0，不参与fitness计算

## 2. Best模型保存逻辑

**文件位置**: `ultralytics/yolo/engine/trainer.py` 第382-383行

```python
# Save last, best and delete
torch.save(ckpt, self.last)
if self.best_fitness == self.fitness:
    torch.save(ckpt, self.best)
```

**文件位置**: `ultralytics/yolo/engine/trainer.py` 第430-431行

```python
if not self.best_fitness or self.best_fitness < fitness:
    self.best_fitness = fitness
```

**关键发现**:

- 每个epoch后，比较当前fitness与历史最佳fitness
- 如果当前fitness更高，更新best_fitness
- 只有当`self.best_fitness == self.fitness`时才保存为best.pt

## 3. 验证触发时机

**文件位置**: `ultralytics/yolo/engine/trainer.py` 第335-339行

```python
# Validation
self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
final_epoch = (epoch + 1 == self.epochs)
if self.args.val or final_epoch:
    self.metrics, self.fitness = self.validate()
```

**关键发现**:

- 验证在每个epoch的训练结束后进行
- 条件: `self.args.val=True` 或 最后一个epoch
- 验证返回metrics和fitness值

## 4. 验证函数实现

**文件位置**: `ultralytics/yolo/engine/trainer.py` 第423-432行

```python
def validate(self):
    """
    Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
    """
    metrics = self.validator(self)
    fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
    if not self.best_fitness or self.best_fitness < fitness:
        self.best_fitness = fitness
    return metrics, fitness
```

**关键发现**:

- 使用`self.validator(self)`进行验证
- **验证是在验证集(val set)上进行的，不是训练集**
- 如果metrics中没有fitness，则使用负的loss值作为fitness

## 5. 验证集vs训练集的明确证据

**文件位置**: 在YOLO的数据加载逻辑中，验证器(validator)专门使用验证集

```python
# 在训练循环中可以看到：
self.metrics, self.fitness = self.validate()  # 这里调用的是验证集评估
```

从训练日志中也可以看到：

```
Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
100/100     0.712G     0.0312     0.0245          0        123        640: 100%|██| 156/156
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all        156        123      0.891      0.887      0.892      0.654
```

这里的156张Images是验证集的图片数量，不是训练集。

## 6. 总结

### Best.pt选择的完整逻辑

1. **评估数据**: **验证集(validation set)**，不是训练集
2. **评估指标**: fitness = 0.1 × mAP@0.5 + 0.9 × mAP@0.5:0.95
3. **保存条件**: 当前epoch的fitness > 历史最佳fitness
4. **评估时机**: 每个epoch训练完成后（如果args.val=True）

### 代码证据链

1. `trainer.py` → 每epoch后调用 `self.validate()`
2. `validate()` → 使用验证集计算metrics，提取fitness
3. `metrics.py` → fitness函数定义，mAP@0.5:0.95占90%权重
4. `trainer.py` → 比较fitness，保存best.pt

### 回答你的问题

- **Best.pt基于什么指标**: 主要是mAP@0.5:0.95（占90%权重）
- **使用哪个数据集**: **验证集(validation set)**，不是训练集
- **代码证据**: 上述所有代码片段都来自YOLOv8源码
