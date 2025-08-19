import random
import numpy as np
import yaml
import os
import torch

def random_seed(seed=42):
    # 设置Python内置随机数种子
    random.seed(seed)
    # 设置NumPy随机数种子
    np.random.seed(seed)
    # 设置PyTorch CPU随机数种子
    torch.manual_seed(seed)
    # 设置PyTorch GPU随机数种子（单卡）
    torch.cuda.manual_seed(seed)
    # 设置所有GPU卡随机数种子（多卡）
    torch.cuda.manual_seed_all(seed)
    # # 固定cuDNN后端优化（避免非确定性算法）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # # 设置环境变量（避免哈希随机化）
    # os.environ['PYTHONHASHSEED'] = str(seed)

def sampled_indices(probabilities, all_num=100, sample_num=10):
    sampled_indices = np.random.choice(
        all_num, 
        size=sample_num, 
        replace=False,  # 无放回采样
        p=probabilities
    )
    return sorted(sampled_indices.tolist())


class ConfigLoader:
    def __init__(self, file_path='config.yaml'):
        self.file_path = file_path
        self.config = self._load_config()
    
    def _load_config(self):
        try:
            with open(self.file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"Config file not found: {self.file_path}")
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value