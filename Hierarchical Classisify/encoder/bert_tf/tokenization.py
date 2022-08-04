import unicodedata


class BaseTokenize:
    def __init__(self, vocab_file):
        self.vocab = self._load_vocab(vocab_file)

    def _load_vocab(self, file):
        pass

    def convert_tokens_to_ids(self):
        pass

    def convert_ids_to_tokens(self):
        pass



