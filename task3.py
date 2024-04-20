import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk import everygrams
from collections import Counter
import numpy as np

# 初始化模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=100, temperature=1.0, top_k=None, top_p=None):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        attention_mask=attention_mask,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def calculate_ngram_diversity(texts, n=1):
    ngrams = Counter()
    total_count = 0
    for text in texts:
        words = text.split()
        ngrams.update(everygrams(words, min_len=n, max_len=n))
        total_count += len(list(everygrams(words, min_len=n, max_len=n)))
    if total_count == 0:
        return 0
    diversity = len(ngrams) / total_count
    return diversity

# 主题提示
prompt = "In a distant future, humanity has begun to colonize other planets."

# 生成文本并调整不同的参数
settings = [
    {"temperature": 0.7, "top_k": 50, "top_p": None},
    {"temperature": 1.0, "top_k": None, "top_p": None},
    {"temperature": 0.7, "top_k": None, "top_p": 0.92}
]

texts = []
for setting in settings:
    text = generate_text(prompt, temperature=setting["temperature"], top_k=setting["top_k"], top_p=setting["top_p"])
    texts.append(text)
    print(f"Generated with temp={setting['temperature']}, top_k={setting['top_k']}, top_p={setting['top_p']}:\n{text}\n")

# 计算和打印每种设置的多样性
for i, text in enumerate(texts):
    setting = settings[i]
    diversity_1 = calculate_ngram_diversity([text], n=1)
    diversity_2 = calculate_ngram_diversity([text], n=2)
    print(f"Settings: Temperature={setting['temperature']}, Top-K={setting['top_k']}, Top-P={setting['top_p']}")
    print(f"Distinct-1: {diversity_1:.4f}, Distinct-2: {diversity_2:.4f}\n")
