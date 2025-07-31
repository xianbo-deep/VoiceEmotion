import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from lora import loraConfig
import yaml
from Dataset import EmotionDataset
from pretrained import Model
import warnings
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
import numpy as np


# 自定义分类器
class AudioDataCollator:
    def __call__(self, features):
        # 过滤None值
        features = [f for f in features if f is not None]

        if not features:
            print(1)
            return {}

        # 提取各个字段
        input_values = [f["input_values"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 创建批次
        batch = {
            "input_values": torch.stack(input_values),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels)
        }

        return batch


warnings.filterwarnings("ignore")

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class trainer():
    def __init__(self):
        base_model = Model()
        processor = base_model.getProcessor()
        loraconfig = loraConfig()

        # 使用我们的自定义数据整理器
        data_collator = AudioDataCollator()
        self.args = TrainingArguments(
            output_dir=config["training"]["lora_checkpoints"],
            weight_decay=config["training"]["weight_decay"],
            warmup_ratio=config["training"]["warmup_ratio"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            # 启用自动保存后，保存模型的步数要是评估步数的倍数，要保证每次评估之后都能保存，而不是评估到一半保存
            load_best_model_at_end=config["training"]["load_best_model_at_end"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
            num_train_epochs=config["training"]["num_train_epochs"],
            eval_steps=config["training"]["eval_steps"],
            eval_strategy=config["training"]["eval_strategy"],
            save_steps=config["training"]["save_steps"],
            save_strategy=config["training"]["save_strategy"],
            report_to=config["training"]["report_to"],
            logging_steps=config["training"]["logging_steps"],
            logging_dir=config["training"]["logging_dir"],
            learning_rate=config["training"]["learning_rate"],
            fp16=config["training"]["fp16"], # 不能和梯度裁剪混用，使用梯度裁剪这个要设为false
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
            max_grad_norm=config["training"]["max_grad_norm"],
            # 使用调度器
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            # 使用多核心进行数据加载的时候若遇到大张量，数据会截断，造成传入datacollator的input_values和attention_mask丢失，因此要做以下配置，不能更改，否则无法运行
            dataloader_num_workers=0,  # 强制单进程
            dataloader_pin_memory=False,  # 禁用内存锁定
            dataloader_persistent_workers=False,  # 禁用持久worker
            remove_unused_columns=False,  # 不删除未使用的列

        )
        # augment是用来判断是否做数据增强的
        train_data = EmotionDataset('./dataset/emotion.csv', processor=processor, split="train",augment=True)
        test_data = EmotionDataset('./dataset/emotion.csv', processor=processor, split="test",augment=False)
        print(f"Train data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")
        self.trainer = Trainer(
            model=loraconfig.get_peft_model(),
            args=self.args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(config["training"]["patience"])],
        )

    def train(self):
        self.trainer.train()
        best_model = self.trainer.model
        best_model.save_pretrained("./best_model")

    # 评估指标
    # eval_pred 是 Trainer 自动传入的参数，包含模型预测结果和真实标签
    def compute_metrics(self, eval_pred):
        predictions,labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2, 3, 4]
        )

        # 获取类别名称
        emotion_labels = config["dataset"]["emotions"]

        # 构建结果字典
        metrics = {
            "accuracy": accuracy,
            "overall_f1": f1_score(labels, predictions, average='weighted'),
        }

        # 添加每个类别的指标
        for i, emotion in enumerate(emotion_labels):
            metrics.update({
                f"{emotion}_precision": precision[i],
                f"{emotion}_recall": recall[i],
                f"{emotion}_f1": f1[i],
            })

        return metrics

if __name__ == "__main__":
    trainer_instance = trainer()
    trainer_instance.train()