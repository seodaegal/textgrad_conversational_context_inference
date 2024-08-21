import os
import json
import pandas as pd
import subprocess
import platformdirs
import textgrad as tg
from .base import Dataset

class BBUNG(Dataset):
    def __init__(self, task_name: str, root: str=None, split: str="train", *args, **kwargs):
        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        self.root = root
        self.split = split
        self.task_name = task_name
        assert split in ["train", "val", "test"]
        data_path = os.path.join(self.root, self.task_name, f"{split}.csv")
        self.data = pd.read_csv(data_path, index_col=0)
        self.task_description = "You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."
        
    def get_task_description(self):
        return self.task_description
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row['input'], row['target']
    
    def __len__(self):
        return len(self.data)