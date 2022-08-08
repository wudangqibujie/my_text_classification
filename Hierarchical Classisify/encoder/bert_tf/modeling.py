import tensorflow as tf
from encoder.bert_tf.config import BertConfig


class BertModel:
    def __init__(self, config, input_ids, is_traing, input_mask, token_type_ids):
        self.config = config
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        if not is_traing:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        

    def _embedding_lookup(self):
        pass

    def _embedding_additive(self):
        pass

    def _create_input_mask(self):
        pass

    def transformer_block(self):
        pass

