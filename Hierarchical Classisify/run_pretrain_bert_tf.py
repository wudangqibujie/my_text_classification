from encoder.jay_bert_tf.dataset import ReadTFrecord
import tensorflow as tf
from encoder.jay_bert_tf.config import BertConfig
from encoder.jay_bert_tf.modeling import BertModel

tf.logging.set_verbosity(tf.logging.INFO)
max_pred_per_seq = 20
max_seq_length = 200
special_tokens_num = 3
batch_size = 64
learning_rate = 2e-5
num_steps = 200
num_warmup_steps = 100
init_checkpoint = r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\bert_model.ckpt"
output_tfrecords = [f"bert_training_tfrecords/train_{i}.tfrecord" for i in range(5)]

# config = BertConfig.from_json_file("encoder/jay_bert_tf/bert_config.json")
config = BertConfig.from_json_file(r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\bert_config.json")

is_traing = True
input_ids_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
input_mask_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
segment_ids_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length])
masked_lm_positions_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_pred_per_seq])
mask_lm_ids_ph = tf.placeholder(dtype=tf.int64, shape=[None, max_pred_per_seq])
mask_lm_weights_ph = tf.placeholder(dtype=tf.float32, shape=[None, max_pred_per_seq])
next_sentence_labels_ph = tf.placeholder(dtype=tf.int64, shape=[None, 1])

model = BertModel(config=config,
                  is_traing=is_traing,
                  input_ids=input_ids_ph,
                  input_mask=input_mask_ph,
                  token_type_ids=segment_ids_ph,
                  mask_lm_position=masked_lm_positions_ph,
                  mask_lm_ids=mask_lm_ids_ph,
                  mask_lm_weights=mask_lm_weights_ph,
                  next_sentence_labels=next_sentence_labels_ph,
                  init_checkpoint=init_checkpoint,
                  learning_rate=learning_rate,
                  num_steps=num_steps,
                  num_warmup_steps=num_warmup_steps
                  )

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for epoch in range(100):
    readTFrecord = ReadTFrecord(output_tfrecords, batch_size, max_seq_length, max_pred_per_seq)
    total_loss = 0.0
    total_sample_num = 0
    while True:
        readTFrecord.get_batch()
        try:
            for dataset in readTFrecord.datasets:
                batch_sample_info = sess.run(dataset)
                input_ids = batch_sample_info["input_ids"]
                batch_num = input_ids.shape[0]
                input_mask = batch_sample_info["input_mask"]
                masked_lm_ids = batch_sample_info["masked_lm_ids"]
                masked_lm_positions = batch_sample_info["masked_lm_positions"]
                masked_lm_weights = batch_sample_info["masked_lm_weights"]
                next_sent_label = batch_sample_info["next_sent_label"]
                segment_ids = batch_sample_info["segment_ids"]
                tf.logging.info([input_ids.shape, input_mask.shape, masked_lm_ids.shape, masked_lm_positions.shape,
                                 masked_lm_weights.shape, next_sent_label.shape, segment_ids.shape])
                loss, _ = sess.run([model.total_loss, model.train_op],
                                   feed_dict={input_ids_ph: input_ids,
                                              input_mask_ph: input_mask,
                                              segment_ids_ph: segment_ids,
                                              masked_lm_positions_ph: masked_lm_positions,
                                              mask_lm_ids_ph: masked_lm_ids,
                                              mask_lm_weights_ph: masked_lm_weights,
                                              next_sentence_labels_ph: next_sent_label})
                total_loss += batch_num * loss
                total_sample_num += batch_num
            tf.logging.info(f"{epoch}-{total_loss / total_sample_num}")
        except tf.errors.OutOfRangeError:
            break
