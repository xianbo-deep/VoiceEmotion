import torch.nn as nn
import torch
import yaml
import torch.nn.functional as F
from pretrained import Model
with open("config.yaml", encoding='utf-8') as f:
    config = yaml.safe_load(f)


class EmotionModel(nn.Module):
    def __init__(self):
        # 定义网络层
        super().__init__()
        model = Model()
        self.pretrained_model = model.getModel()
        # 添加config属性，从预训练模型复制或创建新的
        self.config = self.pretrained_model.config
        # 确保config有use_return_dict属性
        if not hasattr(self.config, 'use_return_dict'):
            self.config.use_return_dict = True

        hidden_size = self.pretrained_model.config.hidden_size
        # 注意力层
        self.attention_pooling = AttentionPooling(hidden_size)
        # 分类器
        # 短音频 要用更加复杂的模型提取信息
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size), # 防止梯度爆炸 归一化
            nn.Dropout(0.1),
            nn.Linear(hidden_size,256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256,config["model"]["layer"]["num_emotions"]),
        )

    # 前向传播
    def forward(self,**kwargs):
        input_values = None
        if 'input_values' in kwargs and kwargs['input_values'] is not None:
            input_values = kwargs['input_values']
        elif 'input_ids' in kwargs and kwargs['input_ids'] is not None:
            input_values = kwargs['input_ids']
        elif 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
            input_values = kwargs['inputs_embeds']

        if input_values is None:
            raise ValueError("Must provide input_values, input_ids, or inputs_embeds")

        attention_mask = kwargs.get('attention_mask')
        labels = kwargs.get('labels')



        # 数据经过预训练模型的输出
        outputs = self.pretrained_model(input_values,attention_mask=attention_mask)
        # 获取每一帧隐藏层的维度
        hidden_states = outputs.last_hidden_state
        # 加权注意力分数
        scores = self.attention_pooling(hidden_states)
        # 分类结果
        classified = self.classifier(scores)

        # 需要手动计算loss
        loss = None
        if labels is not None:
            # 使用交叉熵
            loss = F.cross_entropy(classified,labels)
        return {
            "loss": loss,
            "logits": classified,
        }


# 注意力加权层 加在整个模型的最后
class AttentionPooling(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size,1),
        )

    ''' 
        x.shape = [B,T,H]
        B:批量大小
        T:一个音频有多少帧
        H:一个帧的特征维度
    '''
    def forward(self,x,attention_mask=None):
        scores = self.layer(x).squeeze(-1) # 移除最后一个维度
        # 对填充的进行分数清零
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        # 用softmax算出每帧的权重
        weights = F.softmax(scores,dim=-1)
        # 每帧乘以对应权重并相加   .unsqueeze()为填充最后一个维度
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, H]
        return pooled
