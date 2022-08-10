import os
import random

folder = r"E:\NLP——project\THUCNews"
# cates = os.listdir(r"E:\NLP——project\THUCNews")
cates = ['体育', '娱乐', '家居', '彩票', '房产']
datasets = []

def read_one_cate(cate):
    cate_folder = os.path.join(folder, cate)
    files = os.listdir(cate_folder)
    debug = 0
    for fl in files:
        if debug % 500 == 0:
            print(debug)
        if debug > 10000:
            break
        f = open(os.path.join(cate_folder, fl), encoding="utf-8")
        raw_text = f.read().split()
        text = "。".join(raw_text)
        sample = "\t".join([cate, text])
        datasets.append(sample)
        f.close()
        debug += 1

for c in cates:
    read_one_cate(c)
    print(c)

random.shuffle(datasets)
total_num = len(datasets)
split_point = int(total_num * 0.8)

train_dataset = datasets[: split_point]
val_dataset = datasets[split_point: ]

print(len(train_dataset), len(val_dataset))

f_train = open(os.path.join(folder, "thu_train.txt"), encoding="utf-8", mode="w")
f_val = open(os.path.join(folder, "thu_val.txt"), encoding="utf-8", mode="w")

for i in train_dataset:
    f_train.write(i + "\n")

for i in val_dataset:
    f_val.write(i + "\n")





