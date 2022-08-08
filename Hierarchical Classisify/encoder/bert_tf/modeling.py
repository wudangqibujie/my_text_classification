import tensorflow as tf
import encoder.bert_tf.utils as utils
from encoder.bert_tf.config import BertConfig


class BertModel:
    def __init__(self, config, input_ids, is_traing, input_mask, token_type_ids, scope=None):
        self.config = config
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.scope = scope
        if not is_traing:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        input_shape = utils.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.variable_scope(self.scope, default_name="bert"):
            with tf.variable_scope("embedding"):
                # [batch,  seq_len, embedding_dim]
                self.embedding_out, self.embedding_table = self._embedding_lookup(input_ids=input_ids,
                                                                                  vocab_size=self.config.vocab_size,
                                                                                  embedding_size=self.config.hidden_size,
                                                                                  initializer_range=self.config.initializer_range,
                                                                                  word_embedding_name="word_embeddings")
                self.embedding_output  = self._embedding_additive(input_tensor=self.embedding_out,
                                                                  use_token_type=True,
                                                                  token_type_ids=token_type_ids,
                                                                  token_type_vocab_size=self.config.vocab_size,
                                                                  token_type_embedding_name="token_type_embeddings",
                                                                  use_position_embedding=True,
                                                                  position_embedding_name="position_embeddings",
                                                                  initializer_range=self.config.initializer_range,
                                                                  max_position_embeddings=self.config.max_position_embeddings,
                                                                  dropout_prob=self.config.hidden_dropout_prob)
            with tf.variable_scope("encoder"):
                attention_mask = self._create_input_mask(input_ids, input_mask) # [batch, seq_len, seq_len]





    def _embedding_lookup(self, input_ids, vocab_size, embedding_size=128, initializer_range=0.02,
                          word_embedding_name="word_embeddings"):
        if input_ids.shaoe.ndim == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])    # [batch, seq_len, 1]
        embedding_table = tf.get_variable(name=word_embedding_name,
                                          shape=[vocab_size, embedding_size],
                                          initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        flat_input_ids = tf.reshape(input_ids, [-1])            # [batch * seq_len, ]
        output = tf.gather(embedding_table, flat_input_ids)     # [batch * seq_len, embedding_dim]
        input_shape = utils.get_shape_list(input_ids)
        output = tf.reshape(output,
                            input_shape[0: -1] + [input_shape[-1] * embedding_size])  # [batch,  seq_len, embedding_dim]
        return output, embedding_table

    def _embedding_additive(self,
                            input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embedding=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
        input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        output = input_tensor
        if use_token_type:
            token_type_table = tf.get_variable(name=token_type_embedding_name,
                                                shape=[token_type_vocab_size, width],
                                                initializer=tf.truncated_normal_initializer(initializer_range))
            flat_token_type_ids = tf.reshape(token_type_ids, [-1]) # [batch * seq_len, ]
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings
        if use_position_embedding:
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=tf.truncated_normal_initializer(initializer_range))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1]) # [seq_len, width]
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
        output = self._layer_norm_and_dropout(output, dropout_prob)
        return output

    def _dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output

    def _layer_norm_and_dropout(self, input_tensor, dropout_prob, name=None):
        output_tensor = tf.contrib.layers.layer_norm(inputs=input_tensor,  begin_norm_axis=-1, begin_params_axis=-1, scope=name)
        output_tensor = self._dropout(output_tensor, dropout_prob)
        return output_tensor

    def _create_input_mask(self, input_ids, input_mask):
        input_shape = utils.get_shape_list(input_ids, expected_rank=[2, 3])
        batch_size = input_shape[0]
        from_seq_length = input_shape[1]

        mask_shape = utils.get_shape_list(input_mask, expected_rank=2)
        to_seq_length = mask_shape[1]

        to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, to_seq_length]), tf.float32)
        broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
        mask = broadcast_ones * to_mask
        return mask   # [batch, seq_len, seq_len]

    def transformer_block(self,
                          input_tensor,
                          attention_mask=None,
                          hidden_size=768,
                          num_hidden_layer=12,
                          num_attention_heads=12,
                          intermediate_size=3072,
                          intermediate_act_fn=utils.gelu,
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          initializer_range=0.02,
                          do_return_all_layers=False):
        assert hidden_size % num_attention_heads == 0
        attention_head_size = int(hidden_size / num_attention_heads) # 64
        input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        assert input_width == hidden_size





