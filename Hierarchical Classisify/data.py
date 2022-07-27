import random
import re

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
        self.vocab_file = vocab_files
        self.word_to_idx = self._word_to_idx()
        self.idx_to_word = self._idx_to_word()
        self.special_num = 2
        self.max_seq_length = 120
        self.vocab_num = len(self.word_to_idx) + self.special_num

    def _word_to_idx(self):
        map_ = dict()
        map_["[UNK]"] = 0
        map_["[PAD]"] = 1
        st = 2
        with open(self.vocab_file, encoding="utf-8") as f:
            for i in f:
                map_[i.strip()] = st
                st += 1
        return map_

    def _idx_to_word(self):
        d = dict()
        for k, v in self.word_to_idx.items():
            d[v] = k

    def clean_text(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def _pad_words(self, ids):
        if len(ids) < self.max_seq_length:
            for _ in range(self.max_seq_length - len(ids)):
                ids.append(self.word_to_idx["[PAD]"])
        else:
            ids = ids[: self.max_seq_length]
        return ids

    def to_ids(self, lines):
        ids = []
        for word in lines:
            if word not in self.word_to_idx:
                ids.append(self.word_to_idx["[UNK]"])
            else:
                ids.append(self.word_to_idx[word])
        return ids

    def tokenize(self, words):
        cleaned = self.clean_text(" ".join(words))
        ids = self.to_ids(cleaned.split())
        ids = self._pad_words(ids)
        return ids


class Multi_H_Dataset:
    def __init__(self, files, cls_std, tokenizer, batch_size=64):
        self.tokenizer = tokenizer
        self.files = files
        self.cls_std = cls_std
        self.batch_size = batch_size
        self.std_label_to_code = [dict() for _ in range(cls_std)]
        self._mark_max_label_code = [-1 for _ in range(cls_std)]
        self.length_info = []

    @property
    def get_label_num(self):
        return self._mark_max_label_code

    def get_batch(self):
        random.shuffle(self.files)
        batch_X, batch_y, batch_text, batch_label_text = [], [], [], []
        for f in self.files:
            for tokens, labels_code, texts, labels_text in self._read_file(f):
                if len(batch_X) == self.batch_size:
                    yield batch_X, batch_y, batch_text, batch_label_text
                    batch_X, batch_y, batch_text, batch_label_text = [], [], [], []
                else:
                    batch_X.append(tokens)
                    batch_text.append(texts)
                    batch_y.append(labels_code)
                    batch_label_text.append(labels_text)
        if len(batch_X) >= int(self.batch_size * 0.5):
            yield batch_X, batch_y, batch_text, batch_label_text

    def _read_file(self, file):
        f = open(file)
        for i in f:
            line = i.strip()
            raw_data = line.strip().split("\t")
            if len(raw_data) < 2:
                continue
            raw_labels, texts = raw_data
            text_ids = self.tokenizer.tokenize(texts)
            labels_text = self._get_labels(raw_labels)
            labels_code = self._label_to_code(labels_text)
            self.length_info.append(len(text_ids))
            yield text_ids, labels_code, [texts], labels_text

    def _label_to_code(self, labels):
        labels_code = []
        for sd in range(self.cls_std):
            std_label = labels[sd]
            if std_label not in self.std_label_to_code[sd]:
                self.std_label_to_code[sd][std_label] = self._mark_max_label_code[sd] + 1
                self._mark_max_label_code[sd] += 1
            labels_code.append(self.std_label_to_code[sd][std_label])
        return labels_code

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
        return real_labels[0]


if __name__ == '__main__':
    tokenizer = Tokenize("vocab.txt")
    dataset = Multi_H_Dataset(["Sample_data/sample_train.txt"], 3, tokenizer)
    for bt_x, bt_y, bt_text, bt_l_t in dataset.get_batch():
        print(bt_x)
        print(bt_y)
        print(bt_text)
        print(bt_l_t)
        print(len(bt_l_t))
    print(dataset.get_label_num)
    print(tokenizer.vocab_num)