import collections
import re

import unicodedata


class BaseTokenize:
    def __init__(self, vocab_file, do_lower_case=True, max_char_num=200):
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.max_char_num = max_char_num
        self.unk_token = '[UNK]'

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
        if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _is_chinese(self, cp):
        if ((0x4E00 <= cp <= 0x9FFF) or
                (0x3400 <= cp <= 0x4DBF) or
                (0x20000 <= cp <= 0x2A6DF) or
                (0x2A700 <= cp <= 0x2B73F) or
                (0x2B740 <= cp <= 0x2B81F) or
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or
                (0x2F800 <= cp <= 0x2FA1F)):
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

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

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
        result = []
        for chr_ in text:
            c = ord(chr_)
            if self._is_chinese(c):
                result.append(" ")
                result.append(chr_)
                result.append(" ")
            else:
                result.append(chr_)
        return "".join(result)

    def _convert(self, tokens, vocab):
        rslt = []
        for t in tokens:
            if t not in vocab:
                print(t)
                rslt.append(vocab["[UNK]"])
            else:
                rslt.append(vocab[t])
        return rslt

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def convert_tokens_to_ids(self, tokens):
        return self._convert(tokens, self.vocab)

    def convert_ids_to_tokens(self, ids):
        return self._convert(ids, self.vocab_inverse)

    def basic_tokenize(self, text):
        text = self._clean_text(text)
        text = self._tokenize_chinese(text)
        raw_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in raw_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._punctuation_process(token))
        out_tokens = self.whitespace_tokenize(" ".join(split_tokens))
        return out_tokens

    def wordpiece_tokenize(self, text):
        output_tokens = []
        for token in self.whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_char_num:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenize(text):
            for sub_token in self.wordpiece_tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens


