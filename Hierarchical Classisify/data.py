import random

# CLS_STD = 3
# f = open("Sample_data/sample_train.txt")
# for i in f:
#     line = i.strip()
#     raw_data = line.strip().split("\t")
#     if len(raw_data) < 2:
#         continue
#     labels, texts = raw_data
#     real_labels = []
#     h_labels = [i.split("@") for i in labels.split(",") if len(i.split("@")) <= CLS_STD]
#     mark = [len(i.split("@")) for i in labels.split(",") if len(i.split("@")) <= CLS_STD]
#     split_idxes = [ix for ix in range(len(mark)) if mark[ix] == 1]
#     split_idxes.append(len(mark))
#     for ix in range(1, len(split_idxes)):
#         real_labels.append(h_labels[split_idxes[ix]-1])
#     for i in real_labels:
#         if len(i) != CLS_STD:
#             for _ in range(CLS_STD - len(i)):
#                 i.append("Null")
#     print(real_labels)
#     print(texts)


class Tokenize:
    def __init__(self, vocab_files):
        pass

    def tokenize(self, words):
        pass


class Multi_H_Dataset:
    def __init__(self, files, cls_std, batch_size=64):
        self.files = files
        self.cls_std = cls_std
        self.batch_size = batch_size

    def get_batch(self):
        random.shuffle(self.files)
        batch_X, batch_y = [], []
        for f in self.files:
            for texts, labels in self._read_file(f):
                if len(batch_X) == self.batch_size:
                    yield batch_X, batch_y
                    batch_X, batch_y = [], []
                else:
                    batch_X.append(texts)
                    batch_y.append(labels)
        if len(batch_X) >= int(self.batch_size * 0.5):
            yield batch_X, batch_y

    def _read_file(self, file):
        f = open(file)
        for i in f:
            line = i.strip()
            raw_data = line.strip().split("\t")
            if len(raw_data) < 2:
                continue
            raw_labels, texts = raw_data
            labels = self._get_labels(raw_labels)
            yield texts, labels

    def _get_labels(self, labels):
        real_labels = []
        h_labels = [i.split("@") for i in labels.split(",") if len(i.split("@")) <= self.cls_std]
        mark = [len(i.split("@")) for i in labels.split(",") if len(i.split("@")) <= self.cls_std]
        split_idxes = [ix for ix in range(len(mark)) if mark[ix] == 1]
        split_idxes.append(len(mark))
        for ix in range(1, len(split_idxes)):
            real_labels.append(h_labels[split_idxes[ix] - 1])
        for i in real_labels:
            if len(i) != self.cls_std:
                for _ in range(self.cls_std - len(i)):
                    i.append("Null")
        return real_labels


if __name__ == '__main__':
    dataset = Multi_H_Dataset(["Sample_data/sample_train.txt"], 2)
    for bt_x, bt_y in dataset.get_batch():
        print(bt_x)
        print(bt_y)
        print(len(bt_y))

