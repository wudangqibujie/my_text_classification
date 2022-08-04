import collections
import unicodedata


class BaseTokenize:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case

    def _load_vocab(self, file):
        vocab = collections.OrderedDict()
        idx = 0
        f = open(file)
        for i in f:
            token = i.strip()
            vocab[token] = idx
            idx += 1
        f.close()
        return vocab

    def _punctuation_process(self, text):
        chars = list(text)
        can_ids = [-1]
        for ix, i in enumerate(chars):
            if self._is_punctuation(i):
                can_ids.append(ix)
        can_ids.append(len(chars))
        rslt = []
        for ix in range(1, len(can_ids)):
            nw, pre = can_ids[ix], can_ids[ix - 1] + 1
            if nw - pre == 0 and nw != len(chars):
                rslt.append(chars[nw])
                continue
            rslt.append("".join(chars[pre: nw]))
            if nw != len(chars):
                rslt.append(chars[nw])
        return [i for i in rslt if i]

    def _is_punctuation(self, char):
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_chinese(self, cp):
        if ((cp >= 0x4E00   and cp <= 0x9FFF)   or
            (cp >= 0x3400   and cp <= 0x4DBF)   or
            (cp >= 0x20000  and cp <= 0x2A6DF)  or
            (cp >= 0x2A700  and cp <= 0x2B73F)  or
            (cp >= 0x2B740  and cp <= 0x2B81F)  or
            (cp >= 0x2B820  and cp <= 0x2CEAF)  or
            (cp >= 0xF900   and cp <= 0xFAFF)   or
            (cp >= 0x2F800  and cp <= 0x2FA1F)):
            return True
        return False

    def _clean_text(self, text):
        rslt = []
        for chr in text:
            c = ord(chr)
            if c == 0 or c == 0xfffd or self._is_control(chr):
                continue
            if self._is_whitespace(chr):
                rslt.append(" ")
            else:
                rslt.append(chr)
        return "".join(rslt)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat in ("Cc", "Cf"):
            return True
        return False

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _tokenize_chinese(self, text):
        rslt = []
        for chr in text:
            c = ord(chr)
            if self._is_chinese(c):
                rslt.append(" ")
                rslt.append(chr)
                rslt.append(" ")
            else:
                rslt.append(chr)
        return "".join(rslt)

    def convert(self, tokens, vocab):
        rslt = []
        for t in tokens:
            if t not in vocab:
                rslt.append(vocab["[UNK]"])
            else:
                rslt.append(vocab[t])
        return rslt

    def convert_tokens_to_ids(self, tokens):
        return self.convert(tokens, self.vocab)

    def convert_ids_to_tokens(self, ids):
        return self.convert(ids, self.vocab_inverse)



