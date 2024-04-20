from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/sty/THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/sty/THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()

model = model.eval()
print("=============== input =====================")
input=input()
response, history = model.chat(tokenizer, input, history=[])
print("============= response =================")
print(response)
# response, history = model.chat(tokenizer, "我独自一人", history=history)
# print(response)
