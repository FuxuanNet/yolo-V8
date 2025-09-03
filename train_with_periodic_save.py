import os
import torch
from pathlib import Path


class EpochSaver:
    """自定义回调类，用于每N个epoch保存模型"""

    def __init__(self, save_interval=10, save_dir=None):
        self.save_interval = save_interval
        self.save_dir = save_dir

    def on_fit_epoch_end(self, trainer):
        """在每个epoch结束时调用"""
        epoch = trainer.epoch

        # 每save_interval个epoch保存一次
        if (epoch + 1) % self.save_interval == 0:
            if self.save_dir is None:
                save_dir = trainer.save_dir / "weights"
            else:
                save_dir = Path(self.save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)

            # 构建保存路径
            save_path = save_dir / f"epoch{epoch + 1}.pt"

            # 创建检查点
            ckpt = {
                "epoch": epoch,
                "model": trainer.model.state_dict(),
                "ema": trainer.ema.ema.state_dict() if trainer.ema else None,
                "updates": trainer.ema.updates if trainer.ema else None,
                "optimizer": trainer.optimizer.state_dict(),
                "best_fitness": trainer.best_fitness,
                "train_args": trainer.args,
            }

            # 保存模型
            torch.save(ckpt, save_path)
            print(f"已保存epoch {epoch + 1}模型到: {save_path}")


def train_model_with_periodic_save():
    """带有定期保存功能的训练函数"""
    from ultralytics import YOLO

    model_path = "yolov8m.pt"
    yaml_path = "E:\\programming\\yolo-V8\\ultralytics\\models\\v8\\yolov8m.yaml"

    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n使用GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f}GB")
    else:
        print("\n警告：未检测到GPU，将使用CPU训练（训练速度会很慢）")

    model = YOLO(yaml_path)

    # 创建自定义保存器
    epoch_saver = EpochSaver(save_interval=10)

    # 添加回调函数
    model.add_callback("on_fit_epoch_end", epoch_saver.on_fit_epoch_end)

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
        print(f"  - epoch10.pt, epoch20.pt...: 每10个epoch的检查点模型")
        print(f"训练日志和指标图表保存在: runs/train/exp9/")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        raise e


if __name__ == "__main__":
    train_model_with_periodic_save()
