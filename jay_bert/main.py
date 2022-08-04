import tensorflow as tf
import modeling
import optimization


input_file = "tf_examples.tfrecord"
output_dir = "pretraining_output"
do_train = True
do_eval = True
bert_config_file = "bert_config.json"
init_checkpoint = ""
train_batch_size = 8
eval_batch_size = 4
max_seq_length = 128
max_predictions_per_seq = 20
num_train_steps = 2
num_warmup_steps = 2
learning_rate = 2e-5


