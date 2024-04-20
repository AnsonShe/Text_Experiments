# import os
# os.environ["HF_ENDPOINT"]="https://hf-mirror.com/"
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# 加载预训练模型和分词器
print("================load model==============")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
         
# 构造prompt
print("============= Input Topic ==============")
prompt = "In a distant future, humanity has begun to colonize other planets."
print(prompt)
# 编码并生成文本
print("===============start encoder======================")
inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device) # 创建一个全为1的attention_mask


outputs = model.generate(inputs,pad_token_id=tokenizer.pad_token_id, max_length=300, do_sample=True,attention_mask=attention_mask)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 输出生成文本
print("=========================Generated Text====================")
print(generated_text)


# 使用不同的温度生成文本
outputs_low_temp = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
outputs_high_temp = model.generate(inputs, max_length=100, do_sample=True, temperature=1.5)

# 使用Top-K采样
outputs_top_k = model.generate(inputs, max_length=100, do_sample=True, top_k=50)

# 使用Top-P采样
outputs_top_p = model.generate(inputs, max_length=100, do_sample=True, top_p=0.92)
