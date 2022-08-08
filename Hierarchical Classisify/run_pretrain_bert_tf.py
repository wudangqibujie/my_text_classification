from encoder.bert_tf.dataset import ReadTFrecord
import tensorflow as tf
from encoder.bert_tf.config import BertConfig


tf.logging.set_verbosity(tf.logging.INFO)
max_pred_per_seq = 20
max_seq_length = 200
special_tokens_num = 3
batch_size = 64
output_tfrecords = [f"bert_training_tfrecords/train_{i}.tfrecord" for i in range(5)]
readTFrecord = ReadTFrecord(output_tfrecords, batch_size, max_seq_length, max_pred_per_seq)

config = BertConfig.from_json_file("encoder/bert_tf/bert_config.json")

sess = tf.Session()
readTFrecord.get_batch()
while True:
    try:
        for dataset in readTFrecord.datasets:
            batch_sample_info = sess.run(dataset)
            input_ids = batch_sample_info["input_ids"]
            input_mask = batch_sample_info["input_mask"]
            masked_lm_ids = batch_sample_info["masked_lm_ids"]
            masked_lm_positions = batch_sample_info["masked_lm_positions"]
            masked_lm_weights = batch_sample_info["masked_lm_weights"]
            next_sent_label = batch_sample_info["next_sent_label"]
            segment_ids = batch_sample_info["segment_ids"]
            tf.logging.info([input_ids.shape, input_mask.shape, masked_lm_ids.shape, masked_lm_positions.shape, masked_lm_weights.shape, next_sent_label.shape, segment_ids.shape])
    except tf.errors.OutOfRangeError:
        break

