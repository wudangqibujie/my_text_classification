

CLS_STD = 3
f = open("Sample_data/sample_train.txt")
for i in f:
    line = i.strip()
    raw_data = line.strip().split("\t")
    if len(raw_data) < 2:
        continue
    labels, texts = raw_data
    real_labels = []
    h_labels = [i.split("@") for i in labels.split(",") if len(i.split("@")) <= CLS_STD]
    mark = [len(i.split("@")) for i in labels.split(",") if len(i.split("@")) <= CLS_STD]
    split_idxes = [ix for ix in range(len(mark)) if mark[ix] == 1]
    split_idxes.append(len(mark))
    for ix in range(1, len(split_idxes)):
        real_labels.append(h_labels[split_idxes[ix]-1])
    for i in real_labels:
        if len(i) != CLS_STD:
            for _ in range(CLS_STD - len(i)):
                i.append("Null")
    print(real_labels)
    print(texts)
