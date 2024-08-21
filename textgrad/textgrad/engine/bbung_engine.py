try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("If you'd like to use bllossom models, please install first")



import argparse
import torch

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union
from .base import EngineLM, CachedEngine


class ChatBBUNG (EngineLM, CachedEngine) :
    SYSTEM_PROMPT = "You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."

    def __init__(
        self,
        model_string="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        system_prompt=SYSTEM_PROMPT,
        device="cuda" if torch.cuda.is_available() else "cpu"
        ):
        
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_bbung_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.model_name = model_string
        self.system_prompt = system_prompt
        self.device = device

        # 
        self.model = AutoModelForCausalLM.from_pretrained(self.model_string, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        assert isinstance(self.system_prompt, str)

    """@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)"""
    

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(
        self, prompt, system_prompt=None, temperature=0.7, max_tokens=200, top_p=0.95
    ):
        
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt


        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        full_prompt = sys_prompt_arg + "\n" + prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        self._save_cache(sys_prompt_arg + prompt, response)
        return response


#llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125") -> model = tg.BlackboxLLM(llm_api_test, system_prompt)