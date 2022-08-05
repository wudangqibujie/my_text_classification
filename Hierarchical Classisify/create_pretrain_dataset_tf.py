import tensorflow as tf
from encoder.bert_tf.tokenization import BaseTokenize


vocab_file = "encoder/bert_tf/vocab.txt"
do_lower = True
max_char_per_word = 20
tf.logging.set_verbosity(tf.logging.INFO)
tokenizer = BaseTokenize(vocab_file, do_lower, max_char_per_word)
corpus_files = ["corpus/sample_text.txt"]

f = open(corpus_files[0], encoding="utf-8")
print(f.readlines())



