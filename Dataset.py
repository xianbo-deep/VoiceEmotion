import random
from torch.utils.data import Dataset
import torch
import pandas as pd
import torchaudio

class EmotionDataset(Dataset):
    def __init__(self,csv_path,processor,split="train",augment=False):
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        self.data = self.data[self.data["split"] == split].reset_index(drop=True) # 重置索引
        self.processor = processor # 加载处理器

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            # waveform为音频波形数据，pt张量
            # sr为采样率
            waveform, sr = torchaudio.load(row['path'])
            # 对少于2s的音频进行数据增强
            if self.augment and waveform.size(1) < 2 * sr:
                waveform = self.strength(waveform, sr)
            inputs = self.processor(
                waveform.squeeze(), # 移除长度为1的维度
                sampling_rate=sr,
                return_tensors="pt",
                # 根据批次动态填充
                padding="max_length",
                truncation=True,
                max_length= 16000 * 5,
            )

            return {
                "input_values": inputs["input_values"].squeeze(0).float(),
                "attention_mask": inputs["attention_mask"].squeeze(0).long(),
                "labels": torch.tensor(row['label'], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {row['path']}: {str(e)}")
            return None

        # 音频增强，对小于2s的音频进行数据增强
    def strength(self,waveform,sample_rate):
        # 时间拉伸随机参数
        stretch_rate = random.uniform(0.8,1.2)
        # 音高随机参数
        pitch_steps = random.randint(-3,3)
        # 应用时间拉伸
        if abs(stretch_rate - 1.0) > 0.05:  # stretch_rate太小的话就不增强，避免微小变化
            waveform = self.time_stretch(waveform, sample_rate, stretch_rate)

        # 应用音高变换
        if pitch_steps != 0:
            waveform = self.pitch_shift(waveform, sample_rate, pitch_steps)

        return waveform


    def time_stretch(self,waveform, sample_rate, stretch_rate = 1.0):
        if stretch_rate == 1.0:
            return waveform
        effects = [
            ["speed",str(stretch_rate)], # 改变播放速度
            ["rate",str(sample_rate)],   # 采样率调整回初始值
        ]
        stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )
        return stretched


    def pitch_shift(self,waveform, sample_rate, pitch_shift_steps):
        if pitch_shift_steps == 0:
            return waveform
            # 计算对应的频率比
        pitch_shift = pitch_shift_steps * 100  # 每半音=100音分
        effects = [
            ["pitch", str(pitch_shift)],# 改变音高
            ["rate", str(sample_rate)]  # 采样率调整回初始值
        ]
        shifted, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects
        )
        return shifted