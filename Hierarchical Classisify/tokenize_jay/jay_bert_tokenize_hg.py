from transformers import BertTokenizer, BertModel
MODEL_NAME = 'hfl/chinese-bert-wwm'


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
tokened = tokenizer("我们都是中国人", return_tensors='pt')
print(tokened)
print(tokenizer.convert_ids_to_tokens(tokened["input_ids"][0]))

out = model(input_ids=tokened["input_ids"])
print(out)

