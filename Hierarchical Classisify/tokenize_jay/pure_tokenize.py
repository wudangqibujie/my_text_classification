import jieba
import json


class Corpus:
    def __init__(self, corpus_files, vocab_file, stopwords_file=None, min_count=10):
        self.corpus_files = corpus_files
        self.stopwords = self.read_stopword(stopwords_file)
        self.word_count = dict()
        self.vocab_file = vocab_file
        self.min_count = min_count
        self.word_2_idx = {"[PAD]": 0, "[UNK]": 1}

    def update_vocab(self, words):
        for word in words:
            if word not in self.word_count:
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1

    def build_vocab(self):
        for corpus_file in self.corpus_files:
            with open(corpus_file, encoding="utf-8") as f:
                for line in f:
                    cleaned_line = self.parse_line(line)
                    if not cleaned_line:
                        continue
                    self.update_vocab(cleaned_line)
        st_idx = len(self.word_2_idx)
        for k, v in self.word_count.items():
            if v <= self.min_count:
                continue
            self.word_2_idx[k] = st_idx
            st_idx += 1

    def write_vocab(self):
        with open(self.vocab_file, "w") as f:
            json.dump(self.word_2_idx, f)

    def parse_line(self, line):
        line = line.strip()
        if line:
            slice_words = [i for i in jieba.cut(line) if i not in self.stopwords]
            return slice_words

    def read_stopword(self, file):
        with open(file, encoding="utf-8") as f:
            data = f.readlines()
            data = [i.strip() for i in data]
        return data


class BaseTokenizer:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file