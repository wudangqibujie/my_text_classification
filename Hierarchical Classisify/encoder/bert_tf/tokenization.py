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


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False
text = "asdasd,dwdwdq!!as,,dw！！qdqw。"
chars = list(text)
i = 0
start_new_word = True
output = []
while i < len(chars):
  char = chars[i]
  if _is_punctuation(char):
    output.append([char])
    start_new_word = True
  else:
    if start_new_word:
      output.append([])
    start_new_word = False
    output[-1].append(char)
  i += 1
print(["".join(x) for x in output])


def func(n):
    if n <= 1:
        print(n)
        return n
    else:
        return func(n-1) + func(n -2)

print(func(8))
print(1+2+3+4+5+6+7+8)