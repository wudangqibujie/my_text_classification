from encoder.bert_tf.tokenization import BaseTokenize
from encoder.bert_tf.modeling import BertModelClassify
import collections
import tensorflow as tf
from encoder.bert_tf.config import BertConfig

tf.logging.set_verbosity(tf.logging.INFO)


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


bert_config = BertConfig.from_json_file(
    r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\bert_config.json")
output_checkpoint = r"D:\NLP_project\chinese_checkpoint"
vocab_file = r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\vocab.txt"
init_checkpoint = r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\bert_model.ckpt"


def read_dataset(file, flg):
    f = open(file, encoding="utf-8")
    data = []
    cnt = 0
    for i in f:
        label, text = i.strip().split("@@@")
        data.append(InputExample(guid=f"{flg}_{cnt}", text_a=text, label=label))
        cnt += 1
    f.close()
    return data


tran_batch_size = 64
num_train_epochs = 10
warmup_proportion = 0.1
max_seq_length = 300
train_dataset = read_dataset(r"data/toutiao_train.txt", "train")
val_dataset = read_dataset(r"data/toutiao_val.txt", "val")
train_tf_file = r"data/toutiao_train.tf_record"
val_tf_file = r"data/toutiao_val.tf_record"

label_list = ['news_culture', 'news_story', 'news_entertainment', 'news_military', 'stock', 'news_sports', 'news_edu',
              'news_tech', 'news_travel', 'news_house', 'news_agriculture', 'news_game', 'news_world', 'news_finance',
              'news_car']

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
    # tf.logging.info(f"**************** Training Sample ****************")
    # tf.logging.info(f"tokens: {tokens}")
    # tf.logging.info(f"input_ids: {input_ids}")
    # tf.logging.info(f"input_mask: {input_mask}")
    # tf.logging.info(f"segment_ids: {segment_ids}")
    # tf.logging.info(f"label_id: {label_id}")
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
        features["is_real_example"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list([feature.is_real_example])))
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        write.write(tf_example.SerializeToString())
        if ix % 10000 == 0:
            tf.logging.info(ix)
    write.close()


def readTfDataset(tf_file):
    def parse(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "label_id": tf.FixedLenFeature([], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }
        features = tf.parse_single_example(record, name_to_features)
        return features

    dataset = tf.data.TFRecordDataset(tf_file)
    dataset = dataset.map(parse).shuffle(200).batch(batch_size=tran_batch_size)
    if "train" in tf_file:
        dataset = dataset.repeat(100)
    dataset = dataset.make_one_shot_iterator().get_next()
    return dataset


def create_model(input_ids, input_mask, segment_ids, labels):
    model = BertModelClassify(config=bert_config,
                              is_traing=True,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              token_type_ids=segment_ids,
                              init_checkpoint=init_checkpoint,
                              learning_rate=5e-5,
                              num_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps)
    output_layer = model.pool_output
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights",
        [len(label_list), hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias",
        [len(label_list)],
        initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits=logits, dim=-1)
        log_probs = tf.nn.log_softmax(logits, dim=-1)
        onehot_labels = tf.one_hot(labels, depth=len(label_list), dtype=tf.float32)
        per_sample_loss = -tf.reduce_sum(onehot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_sample_loss)
        train_op = model.create_optimizer(loss, init_learning_rate=5e-5, num_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps)
        return loss, per_sample_loss, logits, probabilities, train_op


input_ids_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
input_mask_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
segment_id_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
labels_ph = tf.placeholder(dtype=tf.int64, shape=[None])

total_loss, per_sample_loss, logits, probabilities, train_op = create_model(input_ids_ph,
                                                                            input_mask_ph,
                                                                            segment_id_ph,
                                                                            labels_ph)
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

tvars = tf.trainable_variables()
assignment_map, initialized_variable_names = BertModelClassify.get_assignment_map_from_checkpoint(tvars,
                                                                                                  init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
for var in tvars:
    init_string = ""
    if var.name and initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)


train_tfdataset = readTfDataset(train_tf_file)
val_tfdataset = readTfDataset(val_tf_file)

while True:
    try:
        train_sample = sess.run(train_tfdataset)
        input_ids = train_sample["input_ids"]
        input_mask = train_sample["input_mask"]
        label_id = train_sample["label_id"]
        segment_ids = train_sample["segment_ids"]

        rs_total_loss, rs_per_sample_loss, rs_logits, rs_probabilities, _ = sess.run(
            [total_loss, per_sample_loss, logits, probabilities, train_op], feed_dict={input_ids_ph: input_ids,
                                                                                       input_mask_ph: input_mask,
                                                                                       labels_ph: label_id,
                                                                                       segment_id_ph: segment_ids})

        tf.logging.info([input_ids.shape, input_mask.shape, label_id.shape, segment_ids.shape])
        tf.logging.info([rs_total_loss, rs_per_sample_loss])
    except tf.errors.OutOfRangeError:
        break
