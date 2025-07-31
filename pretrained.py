import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import Wav2Vec2Model,Wav2Vec2Processor

import yaml
with open("config.yaml", encoding ="utf-8") as f:
    config = yaml.safe_load(f)

class Model:
    def __init__(self):
        model_name = config["model"]["model_name"]
        self.processor = Wav2Vec2Processor.from_pretrained(model_name,return_attention_mask=True)
        self.model =Wav2Vec2Model.from_pretrained(model_name)


    def getModel(self):
        return self.model
    def getProcessor(self):
        return self.processor
