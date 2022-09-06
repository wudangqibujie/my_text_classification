from encoder.jay_bert_tf.tokenization import BaseTokenize
from encoder.jay_bert_tf.dataset import Corpus, Doc2Sample, WriteCorpusTFDataset
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.DEBUG)
max_pred_per_seq = 20
max_seq_length = 200
special_tokens_num = 3
tokenize = BaseTokenize(r"D:\NLP_project\chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12\vocab.txt")


corpus = Corpus("corpus/chinese_sample.txt", tokenize)
corpus.extract_docs()
output_tfrecords = [f"bert_training_tfrecords/train_{i}.tfrecord" for i in range(5)]
for doc_ix, d in enumerate(corpus.documents):
    doc2Sample = Doc2Sample(doc_ix, corpus.documents, max_seq_length, special_tokens_num, list(tokenize.vocab.keys()))
    for _ in range(5):
        doc2Sample.create_sent_pairs()
    for ins in doc2Sample.instances:
        tf.logging.info(ins)
    writeCorpusTFDataset = WriteCorpusTFDataset(doc2Sample.instances,
                                                output_tfrecords,
                                                tokenize,
                                                max_seq_length,
                                                max_pred_per_seq)
    writeCorpusTFDataset.writeTF()
    writeCorpusTFDataset.cloas_writers()

