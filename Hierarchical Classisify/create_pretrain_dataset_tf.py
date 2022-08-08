from encoder.bert_tf.tokenization import BaseTokenize
from encoder.bert_tf.dataset import Corpus, Doc2Sample, WriteCorpusTFDataset
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)
max_pred_per_seq = 20
max_seq_length = 200
special_tokens_num = 3
tokenize = BaseTokenize("encoder/bert_tf/vocab.txt")
corpus = Corpus("corpus/sample_text.txt", tokenize)
corpus.extract_docs()
output_tfrecords = [f"bert_training_tfrecords/train_{i}.tfrecord" for i in range(5)]
for doc_ix, d in enumerate(corpus.documents):
    doc2Sample = Doc2Sample(doc_ix, corpus.documents, max_seq_length, special_tokens_num, list(tokenize.vocab.keys()))
    for _ in range(5):
        doc2Sample.create_sent_pairs()
    writeCorpusTFDataset = WriteCorpusTFDataset(doc2Sample.instances,
                                                output_tfrecords,
                                                tokenize,
                                                max_seq_length,
                                                max_pred_per_seq)
    writeCorpusTFDataset.writeTF()
    writeCorpusTFDataset.cloas_writers()

