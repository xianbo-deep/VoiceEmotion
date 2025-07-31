from peft import get_peft_model,LoraConfig,TaskType
from model import EmotionModel
import yaml

with open("config.yaml", encoding ='utf-8') as f:
    config = yaml.safe_load(f)

class loraConfig():
    def __init__(self):
        self.loraconfig = LoraConfig(
            r = config["loraconfig"]["r"],
            lora_alpha = config["loraconfig"]["lora_alpha"],
            lora_dropout=config["loraconfig"]["lora_dropout"],
            bias = config["loraconfig"]["bias"],
            task_type=TaskType.SEQ_CLS,
            target_modules = config["loraconfig"]["target_modules"],
        )

    def get_peft_model(self):
        model = EmotionModel()
        peft_model = get_peft_model(model,self.loraconfig)
        peft_model.print_trainable_parameters()
        return peft_model