from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json
import random


class DataSet:
    def __init__(self, base_dir_, label_file, max_label_len=20, batch_size=64, split_rt=0.8):
        self.base_file_ = base_dir_
        self.max_label_len = max_label_len
        self.data_info = self._get_label_info(base_dir_, label_file)
        self.batch_size = batch_size
        self.split_rt = split_rt
        self.vocab = self.load_vocab("vocab.json")

    def _get_label_info(self, base_dir_, label_file):
        f = open(os.path.join(base_dir_, label_file), encoding="utf-8")
        lines = f.readlines()
        lines = [i.strip().split("\t") for i in lines]
        return lines

    def build_vocab(self):
        word_to_ids = dict()
        for d_info in self.data_info:
            words = d_info[-1]
            for word in words:
                if word in word_to_ids:
                    continue
                word_to_ids[word] = len(word_to_ids)
        with open("vocab.json", "w") as f:
            json.dump(word_to_ids, f)

    def load_vocab(self, file):
        # 1229
        with open(file) as f:
            word_to_idx = json.load(f)
        return word_to_idx

    def split_data_info(self):
        random.shuffle(self.data_info)
        split_idx = int(len(self.data_info) * self.split_rt)
        return self.data_info[: split_idx], self.data_info[split_idx: ]

    def parse_data_info(self, data_info):
        X, y = [], []
        for d_info in data_info:
            im_file, label_info = d_info
            im = Image.open(os.path.join(self.base_file_, im_file))
            im_dat = np.array(im)
            X.append(im_dat)
            if len(label_info) > self.max_label_len:
                label_info = label_info[: self.max_label_len]
            else:
                for _ in range(self.max_label_len - len(label_info)):
                    label_info += " "
            y.append([self.vocab[i] for i in label_info])
        return np.array(X) / 255., np.array(y)

    def get_bacth(self, data_info):
        num_batch = len(data_info) // self.batch_size
        for n in range(num_batch + 1):
            batch = data_info[n * self.batch_size: (n + 1) * self.batch_size]
            batch_X, batch_y = self.parse_data_info(batch)
            yield batch_X, batch_y


if __name__ == '__main__':
    base_dir = "../../data/ocr/capture"
    dataSet = DataSet(base_dir, "label.txt")
    train_info, val_info = dataSet.split_data_info()
    for X, y in  dataSet.get_bacth(val_info):
        print(X.shape, y.shape)