import torch
import os

# 指定模型路径
best_pt_path = 'E:\\programming\\yolo-V8\\runs\\train\\exp8\\weights\\best.pt'

# 加载模型文件
print(f"正在加载模型文件: {best_pt_path}")
try:
    checkpoint = torch.load(best_pt_path, map_location='cpu', weights_only=False)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 提取训练轮次信息
epoch_info = None
metrics_info = {}

# 尝试从不同可能的位置获取epoch信息
if 'epoch' in checkpoint:
    epoch_info = checkpoint['epoch']
elif 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'epoch' in checkpoint['model']:
    epoch_info = checkpoint['model']['epoch']
elif 'training_results' in checkpoint and isinstance(checkpoint['training_results'], dict):
    if 'epoch' in checkpoint['training_results']:
        epoch_info = checkpoint['training_results']['epoch']
    # 提取可能的评估指标
    if 'metrics' in checkpoint['training_results']:
        metrics_info = checkpoint['training_results']['metrics']
elif 'train_metrics' in checkpoint:
    if 'epoch' in checkpoint['train_metrics']:
        epoch_info = checkpoint['train_metrics']['epoch']
    metrics_info = checkpoint['train_metrics']

# 输出epoch结果
print(f"\nbest.pt模型对应的训练轮次: {epoch_info}")

# 解释epoch=- 的可能含义
if epoch_info == -1:
    print("\n轮次为-1的可能含义:")
    print("1. 这可能是一个预训练模型，未在您的数据集上进行训练")
    print("2. 模型可能是直接从其他地方加载的，不是通过常规训练过程保存的")
    print("3. 在YOLOv8中，-1有时表示这是最终选择的'最佳'模型，不一定对应某个具体训练轮次")
    print("4. 也可能是模型保存过程中出现了问题，导致epoch信息未正确记录")

# 输出评估指标信息
if metrics_info:
    print("\n模型评估指标:")
    for key, value in metrics_info.items():
        print(f"  {key}: {value}")

# 显示模型文件结构，方便进一步分析
print("\n模型文件包含的键:")
for key in checkpoint.keys():
    print(f"- {key}")
    # 显示一些关键键的详细信息
    if key in ['train_args', 'args'] and isinstance(checkpoint[key], dict):
        print("  训练参数摘要:")
        important_args = ['epochs', 'patience', 'batch', 'imgsz', 'optimizer', 'lr0']
        for arg in important_args:
            if arg in checkpoint[key]:
                print(f"    {arg}: {checkpoint[key][arg]}")

# 检查是否有其他可能包含轮次信息的数据
print("\n检查其他可能包含轮次信息的位置...")
if 'results' in checkpoint and isinstance(checkpoint['results'], list):
    print(f"  results字段包含 {len(checkpoint['results'])} 个元素")
elif 'history' in checkpoint:
    print(f"  发现history字段")
    if isinstance(checkpoint['history'], dict):
        print(f"    history包含的键: {', '.join(checkpoint['history'].keys())}")

print("\n分析完成!")