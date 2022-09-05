import jieba
import json
import collections
import tokenize_jay.utils as utils
from tqdm import tqdm


class Corpus:
    def __init__(self, corpus_files, vocab_file, parse_line=None, stopwords_file=None, min_count=10):
        self.corpus_files = corpus_files
        self.stopwords = self.read_stopword(stopwords_file) if stopwords_file else None
        self.word_count = dict()
        self.vocab_file = vocab_file
        self.min_count = min_count
        self.word_2_idx = {"[PAD]": 0, "[UNK]": 1}
        self.parse_line = parse_line

    def update_vocab(self, words):
        for word in words:
            if word not in self.word_count:
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1

    def build_vocab_per_corpus(self, corpus_file):
        with open(corpus_file, encoding="utf-8") as f:
            for line in f:
                if self.parse_line:
                    line = self.parse_line(line)
                tokens = self.tokenize(line)
                if not tokens:
                    continue
                self.update_vocab(tokens)

    def build_vocab(self):
        for corpus_file in tqdm(self.corpus_files, desc="read_file"):
            self.build_vocab_per_corpus(corpus_file)
        st_idx = len(self.word_2_idx)
        for k, v in self.word_count.items():
            if v <= self.min_count:
                continue
            self.word_2_idx[k] = st_idx
            st_idx += 1

    def write_vocab(self):
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            for word in self.word_2_idx.keys():
                f.write(f"{word}\n")

    def tokenize(self, text):
        line = text.strip()
        if line:
            slice_words = [i for i in jieba.cut(line)]
            return slice_words

    def read_stopword(self, file):
        with open(file, encoding="utf-8") as f:
            data = f.readlines()
            data = [i.strip() for i in data]
        return data

    @classmethod
    def check_vocab(cls, vocab_file):
        pass


class BaseTokenizer:
    def __init__(self, vocab_file, stop_word_file=None):
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.stop_words = utils.read_stopword(stop_word_file) if stop_word_file else None

    def _load_vocab(self, file):
        vocab = collections.OrderedDict()
        idx = 0
        f = open(file, encoding="utf-8")
        for i in f:
            token = i.strip()
            vocab[token] = idx
            idx += 1
        f.close()
        return vocab

    def tok_chinese_word(self, text):
        return list(jieba.cut(text))

    def _convert(self, tokens, vocab):
        rslt = []
        for t in tokens:
            if t not in vocab:
                rslt.append(vocab["[UNK]"])
            else:
                rslt.append(vocab[t])
        return rslt

    def convert_tokens_to_ids(self, tokens):
        return self._convert(tokens, self.vocab)

    def convert_ids_to_tokens(self, ids):
        return self._convert(ids, self.vocab_inverse)

    def tokenize(self, text):
        tokens = self.tok_chinese_word(text)
        if self.stop_words:
            tokens = [i for i in tokens if i not in self.stop_words]
        return tokens


if __name__ == '__main__':
    vocab_file = r'..\..\..\chinese_L-12_H-768_A-12\vocab.txt'
    tokenzier = BaseTokenizer(vocab_file)
    text = "我是一个中国人！"
    tokens = tokenzier.tokenize(text)
    print(tokens)
    ids = tokenzier.convert_tokens_to_ids(tokens)
    print(ids)
    tokened = tokenzier.convert_ids_to_tokens(ids)
    print(tokened)