import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

with open("config.yaml",encoding='utf-8') as f:
    config = yaml.safe_load(f)

data = []
emotions = config["dataset"]["emotions"]
base_path = config["dataset"]["base_path"]


for item in os.listdir(base_path):
    speaker_path = os.path.join(base_path, item)
    if not os.path.isdir(speaker_path) or "__pycache__" in speaker_path:
        continue
    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path,emotion)
        if not os.path.isdir(emotion_path):
            continue
        for audio in os.listdir(emotion_path):
            if audio[-4:] == '.wav':
                audio_path = os.path.join(emotion_path,audio)
                data.append(
                    {
                        "path": audio_path,
                        "label": emotion,
                    }
                )


# 转换为df格式
data = pd.DataFrame(data)
# 获取{label:idx}
# label:idx是存储的格式
# 后面的for循环是查询label和idx，先去重，再排序，最后为处理好的label分配连续索引
label2id = {label: idx for idx, label in enumerate(sorted(set(data['label'])))}

# 把label改为数字索引
data['label'] = data['label'].map(label2id)

# 分割数据集测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# 新增split列
train_data["split"] = "train"
test_data["split"] = "test"


# 合并并打乱顺序
final_data = pd.concat([train_data, test_data]).sample(frac=1).reset_index(drop=True)
final_data.to_csv("./dataset/emotion.csv", index=False)

# 保存标签映射
# 两列 label和id
pd.DataFrame({"label": list(label2id.keys()),"id": list(label2id.values())}).to_csv("./dataset/label.csv", index=False)