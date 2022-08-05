


class Corpus:
    def __init__(self, corpus_file):
        self.corpus_fp = open(corpus_file, encoding="utf-8")

    def extract_docs(self):
        pass


class Doc2Sample:
    def __init__(self, doc):
        self.doc = doc

    def create_sen_pair(self):
        pass

    def create_mask(self):
        pass

    def create_next_sen(self):
        pass


class TraingSample:
    def __init__(self, sample_info):
        self.sample_info = sample_info


class WriteCorpusDataset:
    def __init__(self, sample):
        self.sample = sample


