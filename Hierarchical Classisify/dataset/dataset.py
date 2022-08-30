import tensorflow as tf
import json
import collections
import jieba


# tf.string_to_hash_bucket_strong()
# tf.string_to_hash_bucket_fast()
# tf.string_to_hash_bucket()

# a = tf.convert_to_tensor([["A", "B"], ["C", "D"]], dtype=tf.string)
# sess = tf.Session()
# print(sess.run(a))
# out = tf.string_to_hash_bucket(a, num_buckets=20)
# print(sess.run(out))
#
# c = tf.convert_to_tensor([["A", "B"], ["C", "D"]], dtype=tf.string)
# print(sess.run(tf.string_to_hash_bucket(c, num_buckets=20)))


class FastTextDataSet:
    def __init__(self):
        pass

    @staticmethod
    def create_vocab(vocab_file, corpus_file, parse_line_func, min_count=10, stop_word_file=None):
        vocab = collections.OrderedDict()
        vocab["[UNK]"] = 0
        counter = dict()
        with open(corpus_file, encoding="utf-8") as f:
            for corpus_line in f:
                words = [i for i in jieba.cut(parse_line_func(corpus_line))]
                if stop_word_file:
                    stop_words = FastTextDataSet.get_stop_word(stop_word_file)
                    words = [i for i in words if i not in stop_words]
                for word in words:
                    if word not in counter:
                        counter[word] = 1
                        continue
                    counter[word] += 1
        for k, v in counter.items():
            if v <= min_count:
                continue
            vocab[k] = len(vocab) + 1
        with open(vocab_file) as f:
            json.dump(vocab, f)

    @staticmethod
    def get_stop_word(file):
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [i.strip() for i in lines if i]
        return lines

