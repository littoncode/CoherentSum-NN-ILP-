from transformers import BertTokenizer
import torch
from pytorch_transformers import BertModel, BertConfig
from preprocess import *
import nltk
import config
import pickle


Doc =  'During briefing in Geneva, WHO chief Tedros Adhanom Ghebreyesus said that while Omicron variant of COVID-19 does appear to be less severe compared to Delta, it does not mean it should be categorised as mild. "Just like previous variants, Omicron is hospitalising people and it is killing people," he added.'

Summary = 'During briefing in Geneva, WHO chief Tedros Adhanom Ghebreyesus said that while Omicron variant of COVID-19 does appear to be less severe compared to Delta, it does not mean it should be categorised as mild. "Just like previous variants, Omicron is hospitalising people and it is killing people," he added.'


tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
sentence='I really enjoyed this movie a lot.'
tokens=tokenizer.tokenize(sentence)
print(tokens)
tokens = ['[CLS]'] + tokens + ['[SEP]']
T=15


seg_ids=[0 for _ in range(len(tokens))]
sent_ids=tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.tensor(sent_ids).unsqueeze(0)
seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

'''c = BertModel.from_pretrained('bert-base-uncased', cache_dir='/home/kjjose/Downloads')
#hidden_reps, cls_head = c(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
hidden_reps, cls_head = c(token_ids, token_type_ids = seg_ids)
print(cls_head.shape)
print(hidden_reps.shape)'''
