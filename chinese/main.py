import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from model import ROBERTAClassifier
from utils import load_metrics, save_metrics, save_checkpoint, load_checkpoint

data_path = "final.csv"
output_path = "result"


# 读取数据
df = pd.read_csv(data_path)
# df = df.drop(['Unnamed: 0'], axis=1)

encode_label = {'FAKE' : 0, 'REAL' : 1}

df['label'] = df['label'].map(encode_label)

# df['length'] = df['text'].apply(lambda x: len(x.split()))
df['length'] = df['text'].apply(lambda x: len(str(x).split()))

plt.style.use("ggplot")
plt.figure(figsize=(10, 8))
sns.histplot(df['length'], bins=2, kde=True)
plt.title('Frequence of documents of a given length', fontsize=14)
plt.xlabel('length', fontsize=14)
plt.savefig("frequency.png")
plt.show()

df.to_csv("news.csv")


# 固定种子和加速训练
torch.manual_seed(17)
if torch.cuda.is_available():
    device = torch.device('cuda:2')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')

print(device)

# 分词器超参数
tokenizer = RobertaTokenizer.from_pretrained("roberta-base",return_dict=False)
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


# Define columns to read.
label_field = Field(sequential=False, use_vocab=False, batch_first=True)
text_field = Field(use_vocab=False, 
                   tokenize=lambda x: tokenizer.encode(x, truncation=True, max_length=1024), 
                   include_lengths=False, 
                   batch_first=True,
                   fix_length=MAX_SEQ_LEN, 
                   pad_token=PAD_INDEX, 
                   unk_token=UNK_INDEX)

fields = {'text' : ('text', text_field), 'label' : ('label', label_field)}


# 划分数据集 7：2：1
train_data, valid_data, test_data = TabularDataset(path="final.csv", 
                                                   format='CSV', 
                                                   fields=fields, 
                                                   skip_header=False).split(split_ratio=[0.70, 0.2, 0.1], 
                                                                            stratified=True, 
                                                                            strata_field='label')
print("train data length:"+str(len(train_data)))
print("valid data length:"+str(len(valid_data)))
print("test data length:"+str(len(test_data)))

# 数据加载器
train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                               batch_size=BATCH_SIZE,
                                               device=device,
                                               shuffle=True,
                                               sort_key=lambda x: len(x.text), 
                                               sort=True, 
                                               sort_within_batch=False)
test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
print("Data Spliting Finish")

# def pretrain(model, optimizer, train_iter,  valid_iter, scheduler = None,valid_period = len(train_iter),num_epochs = 5):
#     # 预训练线性层，冻结bert
#     for param in model.roberta.parameters():
#         param.requires_grad = False  
    
#     model.train()
#     train_loss = 0.0
#     valid_loss = 0.0   
#     global_step = 0  
#     for epoch in range(num_epochs):
#         for (source, target), _ in train_iter:
#             mask = (source != PAD_INDEX).type(torch.uint8)
#             y_pred = model(input_ids=source, attention_mask=mask)
            
#             loss = torch.nn.CrossEntropyLoss()(y_pred, target)
#             loss.backward()
#             optimizer.step()    
#             scheduler.step()    
#             optimizer.zero_grad()
#             train_loss += loss.item()
#             global_step += 1

#             # print(" Evaluate ")
#             if global_step % valid_period == 0:
#                 model.eval()
#                 with torch.no_grad():                    
#                     for (source, target), _ in valid_iter:
#                         mask = (source != PAD_INDEX).type(torch.uint8)
#                         y_pred = model(input_ids=source,attention_mask=mask)
#                         loss = torch.nn.CrossEntropyLoss()(y_pred, target)
#                         valid_loss += loss.item()
#                 # 保存损失
#                 train_loss = train_loss / valid_period
#                 valid_loss = valid_loss / len(valid_iter)
                            
#                 model.train()
#                 print('Epoch [{}/{}], global step [{}/{}], PT Loss: {:.4f}, Val Loss: {:.4f}'
#                       .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
#                               train_loss, valid_loss))
                
#                 train_loss = 0.0                
#                 valid_loss = 0.0
    
#     # 解冻参数
#     for param in model.roberta.parameters():
#         param.requires_grad = True
#     print('============== 完成线性层预训练 =========')

# def train(model,optimizer,train_iter,valid_iter,scheduler = None,num_epochs = 5,valid_period = len(train_iter),output_path = output_path):
#     train_loss = 0.0
#     valid_loss = 0.0
#     train_loss_list = []
#     valid_loss_list = []
#     best_valid_loss = float('Inf')
#     global_step = 0
#     global_steps_list = []
#     print("================训练骨干模型 =================")
#     model.train()
#     for epoch in range(num_epochs):
#         for (source, target), _ in train_iter:
#             mask = (source != PAD_INDEX).type(torch.uint8)

#             y_pred = model(input_ids=source,  
#                            attention_mask=mask) 
#             loss = torch.nn.CrossEntropyLoss()(y_pred, target)            
#             loss.backward()
#             optimizer.step()    
#             scheduler.step()
#             optimizer.zero_grad()
#             train_loss += loss.item()
#             global_step += 1
#             if global_step % valid_period == 0:
#                 model.eval()
                
#                 with torch.no_grad():                    
#                     for (source, target), _ in valid_iter:
#                         mask = (source != PAD_INDEX).type(torch.uint8)
#                         y_pred = model(input_ids=source, attention_mask=mask)
#                         loss = torch.nn.CrossEntropyLoss()(y_pred, target)
#                         valid_loss += loss.item()
#                 train_loss = train_loss / valid_period
#                 valid_loss = valid_loss / len(valid_iter)
#                 train_loss_list.append(train_loss)
#                 valid_loss_list.append(valid_loss)
#                 global_steps_list.append(global_step)

#                 print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
#                       .format(epoch+1, num_epochs, global_step, num_epochs*len(train_iter),
#                               train_loss, valid_loss))
                
#                 if best_valid_loss > valid_loss:
#                     best_valid_loss = valid_loss
#                     save_checkpoint(output_path + '/model.pkl', model, best_valid_loss)
#                     save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
        
#                 train_loss = 0.0                
#                 valid_loss = 0.0
#                 model.train()    
#     save_metrics(output_path + '/metric.pkl', train_loss_list, valid_loss_list, global_steps_list)
#     print('Training done!')





# NUM_EPOCHS = 6
# steps_per_epoch = len(train_iter)
# model = ROBERTAClassifier(0.4)
# model = model.to(device)
# optimizer = AdamW(model.parameters(), lr=1e-4)
# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps=steps_per_epoch*1, 
#                                             num_training_steps=steps_per_epoch*NUM_EPOCHS)

# print("======================= Start pretraining ==============================")

# pretrain(model=model,
#          train_iter=train_iter,
#          valid_iter=valid_iter,
#          optimizer=optimizer,
#          scheduler=scheduler,
#          num_epochs=NUM_EPOCHS)

# NUM_EPOCHS = 12
# print("======================= Start training =================================")
# optimizer = AdamW(model.parameters(), lr=2e-6)
# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps=steps_per_epoch*2, 
#                                             num_training_steps=steps_per_epoch*NUM_EPOCHS)
# train(model=model, 
#       train_iter=train_iter, 
#       valid_iter=valid_iter, 
#       optimizer=optimizer, 
#       scheduler=scheduler, 
#       num_epochs=NUM_EPOCHS)
# plt.figure(figsize=(10, 8))
# train_loss_list, valid_loss_list, global_steps_list = load_metrics(output_path + '/metric.pkl')
# plt.plot(global_steps_list, train_loss_list, label='Train')
# plt.plot(global_steps_list, valid_loss_list, label='Valid')
# plt.xlabel('Global Steps', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(fontsize=14)
# plt.savefig("loss.png")
# plt.show() 

def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)              
                output = model(source, attention_mask=mask)
                y_pred.extend(torch.argmax(output, axis=-1).tolist())
                y_true.extend(target.tolist())    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

print("============= Start evaluate================")
model = ROBERTAClassifier()
model = model.to(device)
load_checkpoint(output_path + '/model.pkl', model)
evaluate(model, test_iter)
