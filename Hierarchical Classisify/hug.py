from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")

sen = ["Hugging face 起初是一家总部位于纽约的聊天机器人初创服务商", "他们本来打算创业做聊天机器人", "然后在github上开源了一个Transformers库"]
batch = tokenizer(sen, padding=True, truncation=True)
print(batch)
