from encoder.bert_tf.tokenization import BaseTokenize
import collections
import tensorflow as tf
from encoder.bert_tf.config import BertConfig


class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

bert_config = BertConfig.from_json_file(r"E:\NLP——project\uncased_L-12_H-768_A-12\bert_config.json")
output_checkpoint = r"E:\NLP——project\chinese_checkpoint"
vocab_file = r"E:\NLP——project\uncased_L-12_H-768_A-12\vocab.txt"


def read_dataset(file, flg):
    f = open(file, encoding="utf-8")
    data = []
    cnt = 0
    for i in f:
        label, text = i.strip().split("\t")
        data.append(InputExample(guid=f"{flg}_{cnt}", text_a=text, label=label))
        cnt += 1
    f.close()
    return data

tran_batch_size = 64
num_train_epochs = 10
warmup_proportion = 0.1
max_seq_length = 300
train_dataset = read_dataset(r"E:\NLP——project\THUCNews\thu_train.txt", "train")
val_dataset = read_dataset(r"E:\NLP——project\THUCNews\thu_val.txt", "val")
train_tf_file = r"E:\NLP——project\THUCNews\thu_train.tf_record"
val_tf_file = r"E:\NLP——project\THUCNews\thu_val.tf_record"

label_list = ['体育', '娱乐', '家居', '彩票', '房产']
tokenizer = BaseTokenize(vocab_file=vocab_file)
num_train_steps = int(len(train_dataset) / tran_batch_size * num_train_epochs)
num_warmup_steps = int(num_train_steps * warmup_proportion)


def convert_single_example(idx, data, label_list, max_seq_length, tokenizer):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    token_a = tokenizer.tokenize(data.text_a)
    token_b = None
    if len(token_a) > max_seq_length - 2:
        token_a = token_a[0: (max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in token_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[data.label]
    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            is_real_example=True)
    return feature


def writeTFRecord(dataset, output_file):
    write = tf.python_io.TFRecordWriter(output_file)
    for ix, data in enumerate(dataset):
        feature = convert_single_example(ix, data, label_list, max_seq_length, tokenizer)
        features = collections.OrderedDict()
        features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_ids)))
        features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.input_mask)))
        features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature.segment_ids)))
        features["label_id"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list([feature.label_id])))
        features["is_real_example"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list([feature.is_real_example])))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        write.write(tf_example.SerializeToString())
    write.close()


writeTFRecord(train_dataset, train_tf_file)
writeTFRecord(val_dataset, val_tf_file)





