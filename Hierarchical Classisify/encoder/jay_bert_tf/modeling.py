import collections
import re
import tensorflow as tf
import encoder.jay_bert_tf.utils as utils
from encoder.jay_bert_tf.optimization import AdamWeightDecayOptimizer
import math


class BertModelClassify:
    def __init__(self, config, input_ids, is_traing, input_mask, token_type_ids, init_checkpoint, learning_rate, num_steps, num_warmup_steps, scope=None):
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
                self.embedding_output = self._embedding_additive(input_tensor=self.embedding_out,
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
                attention_mask = self._create_input_mask(input_ids, input_mask)  # [batch, seq_len, seq_len]
                self.all_encoder_layers = self.transformer_block(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=self.config.hidden_size,
                    num_hidden_layer=self.config.num_hidden_layers,
                    num_attention_heads=self.config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=utils.get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True
                )
            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0: 1, :], axis=1)
                self.pool_output = tf.layers.dense(
                    first_token_tensor,
                    self.config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(self.config.initializer_range)
                )

    def create_optimizer(self, loss, init_learning_rate, num_steps, num_warmup_steps):
        gloabel_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=init_learning_rate,
                                    shape=[],
                                    dtype=tf.float32)
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            gloabel_step,
            num_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False
        )
        if num_warmup_steps:
            global_steps_int = tf.cast(gloabel_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate
            )
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=gloabel_step)
        new_global_step = gloabel_step + 1
        train_op = tf.group(train_op, gloabel_step.assign(new_global_step))
        return train_op

    @staticmethod
    def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
        initialized_variable_names = {}
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var
        init_vars = tf.train.list_variables(init_checkpoint)
        assignment_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1
        return assignment_map, initialized_variable_names

    def _embedding_lookup(self, input_ids, vocab_size, embedding_size=128, initializer_range=0.02,
                          word_embedding_name="word_embeddings"):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])  # [batch, seq_len, 1]
        embedding_table = tf.get_variable(name=word_embedding_name,
                                          shape=[vocab_size, embedding_size],
                                          initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        flat_input_ids = tf.reshape(input_ids, [-1])  # [batch * seq_len, ]
        output = tf.gather(embedding_table, flat_input_ids)  # [batch * seq_len, embedding_dim]
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
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # [batch * seq_len, ]
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings
        if use_position_embedding:
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=tf.truncated_normal_initializer(initializer_range))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])  # [seq_len, width]
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
        output = self.layer_norm_and_dropout(output, dropout_prob)
        return output

    def dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output

    def layer_norm(self, input_tensor, name=None):
        return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1,
                                            scope=name)

    def layer_norm_and_dropout(self, input_tensor, dropout_prob, name=None):
        output_tensor = tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1,
                                                     scope=name)
        output_tensor = self.dropout(output_tensor, dropout_prob)
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
        return mask  # [batch, seq_len, seq_len]

    def _transpose_for_scores(self,
                              input_tensor,
                              batch_size,
                              num_attention_heads,
                              seq_length,
                              width):
        output_tensor = tf.reshape(input_tensor,
                                   [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):
        from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = utils.get_shape_list(to_tensor, expected_rank=[2, 3])
        assert len(from_shape) == len(to_shape)
        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        from_tensor_2d = utils.reshape_to_matrix(from_tensor)  # [batch_size * seq_length, embedding_dim]
        to_tensor_2d = utils.reshape_to_matrix(to_tensor)  # [batch_size * seq_length, embedding_dim]
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        query_layer = tf.layers.dense(from_tensor_2d,
                                      num_attention_heads * size_per_head,
                                      activation=query_act,
                                      name="query",
                                      kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        key_layer = tf.layers.dense(to_tensor_2d,
                                    num_attention_heads * size_per_head,
                                    activation=key_act,
                                    name="key",
                                    kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        value_layer = tf.layers.dense(to_tensor_2d,
                                      num_attention_heads * size_per_head,
                                      activation=value_act,
                                      name="value",
                                      kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size, num_attention_heads, seq_length, size_per_head]
        query_layer = self._transpose_for_scores(query_layer,
                                                 batch_size,
                                                 num_attention_heads,
                                                 from_seq_length,
                                                 size_per_head)
        # [batch_size, num_attention_heads, seq_length, size_per_head]
        key_layer = self._transpose_for_scores(key_layer,
                                               batch_size,
                                               num_attention_heads,
                                               to_seq_length,
                                               size_per_head)
        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))
        if attention_mask is not None:
            # [batch, 1, seq_len, seq_len]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores *= adder
        # [batch_size, num_attention_heads, from_seq_length, to_seq_length]
        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        value_layer = tf.reshape(value_layer,
                                 [batch_size, to_seq_length, num_attention_heads, size_per_head])
        # [batch, num_attention_heads, to_seq_length, size_per_head]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        # [batch, num_attention_heads, from_seq_length, size_per_head]
        context_layer = tf.matmul(attention_probs, value_layer)
        # [batch, from_seq_length, num_attention_heads, size_per_head]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        if do_return_2d_tensor:
            context_layer = tf.reshape(context_layer,
                                       [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            context_layer = tf.reshape(context_layer,
                                       [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return context_layer

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
        attention_head_size = int(hidden_size / num_attention_heads)  # 64
        input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        assert input_width == hidden_size
        prev_output = utils.reshape_to_matrix(input_tensor)
        all_layer_outputs = []
        for layer_idx in range(num_hidden_layer):
            with tf.variable_scope(f"layer_{layer_idx}"):
                layer_input = prev_output
                with tf.variable_scope(f"attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = self.attention_layer(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length
                        )
                        attention_heads.append(attention_head)
                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        attention_output = tf.concat(attention_heads, axis=-1)

                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=tf.truncated_normal_initializer(initializer_range)
                        )
                        attention_output = self.dropout(attention_output, hidden_dropout_prob)
                        attention_output = self.layer_norm(attention_output + layer_input)
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=tf.truncated_normal_initializer(initializer_range)
                    )
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=tf.truncated_normal_initializer(initializer_range),
                    )
                    layer_output = self.dropout(layer_output, hidden_dropout_prob)
                    layer_output = self.layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)
        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = utils.reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = utils.reshape_from_matrix(prev_output, input_shape)
            return final_output


class BertModel:
    def __init__(self, config, input_ids, is_traing, input_mask, token_type_ids, mask_lm_position, mask_lm_ids,
                 mask_lm_weights, next_sentence_labels, init_checkpoint, learning_rate, num_steps, num_warmup_steps, scope=None):
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
                self.embedding_output = self._embedding_additive(input_tensor=self.embedding_out,
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
                attention_mask = self._create_input_mask(input_ids, input_mask)  # [batch, seq_len, seq_len]
                self.all_encoder_layers = self.transformer_block(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=self.config.hidden_size,
                    num_hidden_layer=self.config.num_hidden_layers,
                    num_attention_heads=self.config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=utils.get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True
                )
            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0: 1, :], axis=1)
                self.pool_output = tf.layers.dense(
                    first_token_tensor,
                    self.config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(self.config.initializer_range)
                )

        mask_lm_loss, mask_lm_sample_loss, mask_lm_log_probs = self.get_masked_lm_output(self.sequence_output,
                                                                                         self.embedding_table,
                                                                                         mask_lm_position,
                                                                                         mask_lm_ids,
                                                                                         mask_lm_weights)
        next_sent_loss, next_sent_sample_loss, next_sent_log_probs = self.get_next_sentence_output(self.pool_output,
                                                                                                   next_sentence_labels)
        total_loss = mask_lm_loss + next_sent_loss
        self.total_loss = total_loss
        tvars = tf.trainable_variables()
        assignment_map, initialized_variable_names = self.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        self.train_op = self.create_optimizer(total_loss,
                                              learning_rate,
                                              num_steps,
                                              num_warmup_steps)

    def create_optimizer(self, loss, init_learning_rate, num_steps, num_warmup_steps):
        gloabel_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=init_learning_rate,
                                    shape=[],
                                    dtype=tf.float32)
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            gloabel_step,
            num_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False
        )
        if num_warmup_steps:
            global_steps_int = tf.cast(gloabel_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_learning_rate * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate
            )
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=gloabel_step)
        new_global_step = gloabel_step + 1
        train_op = tf.group(train_op, gloabel_step.assign(new_global_step))
        return train_op

    def get_assignment_map_from_checkpoint(self, tvars, init_checkpoint):
        initialized_variable_names = {}
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var
        init_vars = tf.train.list_variables(init_checkpoint)
        assignment_map = collections.OrderedDict()
        for x in init_vars:
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1
        return assignment_map, initialized_variable_names

    def gather_indexes(self, sequnce_tensor, positions):
        sequence_shape = utils.get_shape_list(sequnce_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]
        debug = tf.cast(tf.range(0, batch_size, dtype=tf.int32) * seq_length, tf.int64)
        flat_offsets = tf.reshape(debug, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])  # [batch, seq_length] [batch, 1]
        flat_sequence_tensor = tf.reshape(sequnce_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def get_next_sentence_output(self, input_tensor, labels):
        with tf.variable_scope("cls/seq_relationship"):
            outpue_weights = tf.get_variable(
                "outpue_weights",
                shape=[2, self.config.hidden_size],
                initializer=tf.truncated_normal_initializer(self.config.initializer_range)
            )
            output_bias = tf.get_variable(
                "output_bias",
                shape=[2],
                initializer=tf.zeros_initializer()
            )
            logits = tf.matmul(input_tensor, outpue_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, dim=-1)
            labels = tf.reshape(labels, [-1])
            onehot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
            per_sample_loss = -tf.reduce_sum(onehot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_sample_loss)
            return loss, per_sample_loss, log_probs

    def get_masked_lm_output(self, input_tensor, output_weights, positions, label_ids, label_weights):
        input_tensor = self.gather_indexes(input_tensor, positions)
        with tf.variable_scope("cls/predictions"):
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=self.config.hidden_size,
                    activation=utils.get_activation(self.config.hidden_act),
                    kernel_initializer=tf.truncated_normal_initializer(self.config.initializer_range)
                )
                input_tensor = self.layer_norm(input_tensor)
            output_bias = tf.get_variable(
                "output_bias",
                shape=[self.config.vocab_size],
                initializer=tf.zeros_initializer()
            )
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, dim=-1)
            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])
            onehot_labels = tf.one_hot(label_ids, depth=self.config.vocab_size, dtype=tf.float32)
            per_sample_loss = -tf.reduce_sum(log_probs * onehot_labels, axis=[-1])
            numerator = tf.reduce_sum(label_weights * per_sample_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator
        return loss, per_sample_loss, log_probs

    def _embedding_lookup(self, input_ids, vocab_size, embedding_size=128, initializer_range=0.02,
                          word_embedding_name="word_embeddings"):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])  # [batch, seq_len, 1]
        embedding_table = tf.get_variable(name=word_embedding_name,
                                          shape=[vocab_size, embedding_size],
                                          initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        flat_input_ids = tf.reshape(input_ids, [-1])  # [batch * seq_len, ]
        output = tf.gather(embedding_table, flat_input_ids)  # [batch * seq_len, embedding_dim]
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
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])  # [batch * seq_len, ]
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings
        if use_position_embedding:
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=tf.truncated_normal_initializer(initializer_range))
            position_embeddings = tf.slice(full_position_embeddings, [0, 0], [seq_length, -1])  # [seq_len, width]
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings
        output = self.layer_norm_and_dropout(output, dropout_prob)
        return output

    def dropout(self, input_tensor, dropout_prob):
        if dropout_prob is None or dropout_prob == 0.0:
            return input_tensor
        output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
        return output

    def layer_norm(self, input_tensor, name=None):
        return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1,
                                            scope=name)

    def layer_norm_and_dropout(self, input_tensor, dropout_prob, name=None):
        output_tensor = tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1,
                                                     scope=name)
        output_tensor = self.dropout(output_tensor, dropout_prob)
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
        return mask  # [batch, seq_len, seq_len]

    def _transpose_for_scores(self,
                              input_tensor,
                              batch_size,
                              num_attention_heads,
                              seq_length,
                              width):
        output_tensor = tf.reshape(input_tensor,
                                   [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None):
        from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = utils.get_shape_list(to_tensor, expected_rank=[2, 3])
        assert len(from_shape) == len(to_shape)
        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        from_tensor_2d = utils.reshape_to_matrix(from_tensor)  # [batch_size * seq_length, embedding_dim]
        to_tensor_2d = utils.reshape_to_matrix(to_tensor)  # [batch_size * seq_length, embedding_dim]
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        query_layer = tf.layers.dense(from_tensor_2d,
                                      num_attention_heads * size_per_head,
                                      activation=query_act,
                                      name="query",
                                      kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        key_layer = tf.layers.dense(to_tensor_2d,
                                    num_attention_heads * size_per_head,
                                    activation=key_act,
                                    name="key",
                                    kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size * seq_length, num_attention_heads * size_per_head]
        value_layer = tf.layers.dense(to_tensor_2d,
                                      num_attention_heads * size_per_head,
                                      activation=value_act,
                                      name="value",
                                      kernel_initializer=tf.truncated_normal_initializer(initializer_range))
        # [batch_size, num_attention_heads, seq_length, size_per_head]
        query_layer = self._transpose_for_scores(query_layer,
                                                 batch_size,
                                                 num_attention_heads,
                                                 from_seq_length,
                                                 size_per_head)
        # [batch_size, num_attention_heads, seq_length, size_per_head]
        key_layer = self._transpose_for_scores(key_layer,
                                               batch_size,
                                               num_attention_heads,
                                               to_seq_length,
                                               size_per_head)
        # [batch_size, num_attention_heads, seq_length, seq_length]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))
        if attention_mask is not None:
            # [batch, 1, seq_len, seq_len]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores *= adder
        # [batch_size, num_attention_heads, from_seq_length, to_seq_length]
        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs, attention_probs_dropout_prob)

        value_layer = tf.reshape(value_layer,
                                 [batch_size, to_seq_length, num_attention_heads, size_per_head])
        # [batch, num_attention_heads, to_seq_length, size_per_head]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
        # [batch, num_attention_heads, from_seq_length, size_per_head]
        context_layer = tf.matmul(attention_probs, value_layer)
        # [batch, from_seq_length, num_attention_heads, size_per_head]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        if do_return_2d_tensor:
            context_layer = tf.reshape(context_layer,
                                       [batch_size * from_seq_length, num_attention_heads * size_per_head])
        else:
            context_layer = tf.reshape(context_layer,
                                       [batch_size, from_seq_length, num_attention_heads * size_per_head])
        return context_layer

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
        attention_head_size = int(hidden_size / num_attention_heads)  # 64
        input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        assert input_width == hidden_size
        prev_output = utils.reshape_to_matrix(input_tensor)
        all_layer_outputs = []
        for layer_idx in range(num_hidden_layer):
            with tf.variable_scope(f"layer_{layer_idx}"):
                layer_input = prev_output
                with tf.variable_scope(f"attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = self.attention_layer(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length
                        )
                        attention_heads.append(attention_head)
                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        attention_output = tf.concat(attention_heads, axis=-1)

                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=tf.truncated_normal_initializer(initializer_range)
                        )
                        attention_output = self.dropout(attention_output, hidden_dropout_prob)
                        attention_output = self.layer_norm(attention_output + layer_input)
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=tf.truncated_normal_initializer(initializer_range)
                    )
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=tf.truncated_normal_initializer(initializer_range),
                    )
                    layer_output = self.dropout(layer_output, hidden_dropout_prob)
                    layer_output = self.layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)
        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = utils.reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = utils.reshape_from_matrix(prev_output, input_shape)
            return final_output
