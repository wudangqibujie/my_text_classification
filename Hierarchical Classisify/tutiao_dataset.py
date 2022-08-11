import random



data = []
label_set = set()
f = open("data/toutiao_cat_data.txt", encoding="utf-8")
for i in f:
    line = i.strip().split("_!_")
    label = line[2]
    text = "。".join([i for i in line[3: ] if i])
    data.append((label, text))
    label_set.add(label)
print(label_set)
total_num = len(data)
split_pt = int(total_num * 0.8)
train = data[: split_pt]
val = data[split_pt: ]
print(total_num)
train_f = open("data/toutiao_train.txt", encoding="utf-8", mode="w")
val_f = open("data/toutiao_val.txt", encoding="utf-8", mode="w")
for i in train:
    train_f.write("@@@".join(i) + "\n")
for i in val:
    val_f.write("@@@".join(i) + "\n")