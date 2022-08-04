import tensorflow as tf

max_seq_length = 128
max_predictions_per_seq = 20
def parse(record):
    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }
    features = tf.parse_single_example(record, name_to_features)
    return (features["input_ids"],
            features["input_mask"],
            features["segment_ids"],
            features["masked_lm_positions"],
            features["masked_lm_ids"],
            features["masked_lm_weights"],
            features["next_sentence_labels"])


input_file = ["tf_examples.tfrecord"]
dataset = tf.data.TFRecordDataset(input_file)
dataset = dataset.map(parse).make_one_shot_iterator().get_next()

if __name__ == '__main__':
    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(dataset))
            except tf.errors.OutOfRangeError:
                break
