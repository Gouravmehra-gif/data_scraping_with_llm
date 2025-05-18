from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

class TextGenModel:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance = {
                "model": model,
                "tokenizer": tokenizer
            }
        return cls._instance

