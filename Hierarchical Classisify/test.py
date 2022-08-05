from encoder.bert_tf.tokenization import BaseTokenize


tokenize = BaseTokenize("encoder/bert_tf/vocab.txt")

sen = "In this paper, we conduct exhaustive experiments to investigate different fine-tuning methods of BERT on text classification task and provide a general solution for BERT fine-tuning."
print(tokenize.tokenize(sen))

