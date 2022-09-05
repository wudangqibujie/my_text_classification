import tensorflow as tf
import random
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


class DataSet:
    def __init__(self, tokenize, max_length, label_info, output_files):
        self.tokenize = tokenize
        self.writers = [tf.python_io.TFRecordWriter(of) for of in output_files]
        self.max_length = max_length
        self.label_info = label_info
        self.output_files = output_files

    def write_tfrecord(self, ids, label):
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        while len(ids) < self.max_length:
            for _ in range(self.max_length - len(ids)):
                ids.append(self.tokenize.vocab['[PAD]'])
        label_code = self.label_info[label]
        features = collections.OrderedDict()
        features["input_ids"] = self._create_int_feature(ids)
        features["label_code"] = self._create_int_feature([label_code])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        write_idx = random.randint(0, len(self.writers) - 1)
        writer = self.writers[write_idx]
        writer.write(tf_example.SerializeToString())

    def _create_int_feature(self, values):
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feature

    def _create_float_feature(self, values):
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feature

    def cloas_writers(self):
        for w in self.writers:
            w.close()


class TextClfTFRecord:
    def __init__(self, tfrecord_files, batch_size, max_seq_length):
        self.tfrecord_files = tfrecord_files
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.datasets = []

    def _parse(self, record):
        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_code":
                tf.FixedLenFeature([1], tf.int64),
        }
        features = tf.parse_single_example(record, name_to_features)
        return features

    def get_batch(self):
        random.shuffle(self.tfrecord_files)
        for tfrecord_file in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            dataset = dataset.map(self._parse).shuffle(100).batch(self.batch_size)
            dataset = dataset.make_one_shot_iterator().get_next()
            self.datasets.append(dataset)


if __name__ == '__main__':
    for _ in range(10):
        print(random.randint(0, 3))


