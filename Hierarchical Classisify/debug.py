from transformers import AutoTokenizer, BertTokenizer, BertConfig, AutoModelForSequenceClassification, BertPreTrainedModel, BertModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd

model_path = r"D:\NLP_project\pretrain_model\hfl_bert_chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
# inputtext = "今天心情情很好啊，买了很多东西，我特别喜欢，终于有了自己喜欢的电子产品，这次总算可以好好学习了"
# tokenized_text = tokenizer.encode(inputtext)
# input_ids = torch.tensor(tokenized_text).view(-1, len(tokenized_text))
# outputs = model(input_ids)
# print(outputs[0].shape, outputs[1].shape)
# print(tokenized_text)


class TextClassifier(torch.nn.Module):
    def __init__(self, bert_model, bert_config, num_class):
        super(TextClassifier, self).__init__()
        self.bert_model = bert_model
        self.bert_config = bert_config
        self.dropout = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = torch.nn.Linear(bert_config.hidden_size, num_class)

    def forward(self, token_ids):
        out = self.bert_model(token_ids)[1] # 句向量 [batch_size, hidden_size]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


f = open("data/toutiao_cat_data.txt", encoding="utf-8")
label_code = {0: 'news_tech', 1: 'news_story', 2: 'news_house', 3: 'stock', 4: 'news_travel', 5: 'news_world', 6: 'news_culture', 7: 'news_car', 8: 'news_military', 9: 'news_game', 10: 'news_sports', 11: 'news_edu', 12: 'news_agriculture', 13: 'news_finance', 14: 'news_entertainment'}
code_label = {'news_tech': 0, 'news_story': 1, 'news_house': 2, 'stock': 3, 'news_travel': 4, 'news_world': 5, 'news_culture': 6, 'news_car': 7, 'news_military': 8, 'news_game': 9, 'news_sports': 10, 'news_edu': 11, 'news_agriculture': 12, 'news_finance': 13, 'news_entertainment': 14}
NUM_CLASS = len(code_label)
EPOCH = 20

def get_batch(fp, tokenizer=tokenizer, batch_size=64):
    batch_text, batch_y = [], []
    for i in fp:
        raw_data = i.strip().split("_!_")
        label_name = raw_data[2]
        texts = "，".join(raw_data[3:]).lstrip("，")
        batch_y.append(code_label[label_name])
        batch_text.append(texts)
        if len(batch_text) >= batch_size:
            batch_X = tokenizer(batch_text, truncation=True, padding=True, max_length=20)
            yield batch_X, np.array(batch_y)
            batch_text, batch_y = [], []
    batch_X = tokenizer.encode_plus(batch_text, truncation=True, padding=True, max_length=20)
    yield batch_X, np.array(batch_y)


textClassifier = TextClassifier(model, config, NUM_CLASS)
device = torch.device("cuda:9") if torch.cuda.is_available() else "cpu"
textClassifier = textClassifier.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimzer = torch.optim.SGD(textClassifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

TOT = 382688


for epoch in range(EPOCH):
    loss_sum = 0.
    accu = 0
    textClassifier.train()
    cnt = 0
    for step, (bt_X, bt_y) in enumerate(get_batch(f)):
        input_ids = torch.tensor(bt_X["input_ids"])
        bt_y = torch.tensor(bt_y, dtype=torch.long)
        out = textClassifier(input_ids)
        loss = criterion(out, bt_y)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        accu += (out.argmax(1) == bt_y).sum().cpu().data.numpy()
        cnt += input_ids.size()[0]
        TOT -= cnt
        if cnt % 10 * 64 == 0:
            print(TOT, cnt, accu / cnt)
    print(f"{epoch}: {accu / cnt}")


