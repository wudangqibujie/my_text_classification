from dataset.dataset import FastTextDataSet


STOP_WORD_FILE = "data/stopwords.txt"
CORPUS_FILE = "data/toutiao_train.txt"
VOCAB_FILE = "data/toutiao_vocab.json"


def parse_line(line):
    line = line.strip()
    line = line.split("@@@")[-1]
    return line


FastTextDataSet.create_vocab(vocab_file=VOCAB_FILE, corpus_file=CORPUS_FILE, parse_line_func=parse_line, stop_word_file=STOP_WORD_FILE)
