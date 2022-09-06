import tensorflow as tf
import collections
import random

tf.logging.set_verbosity(tf.logging.DEBUG)


class Corpus:
    def __init__(self, corpus_file, tokenizer):
        self.corpus_fp = open(corpus_file, encoding="utf-8")
        self.documents = []
        self.tokenizer = tokenizer

    def extract_docs(self):
        doc = []
        for i in self.corpus_fp:
            if not i.strip():
                self.documents.append(doc)
                doc = []
                continue
            doc.append(self.tokenizer.tokenize(i.strip()))
        self.documents.append(doc)


class Doc2Sample:
    def __init__(self, doc_ix, docs, max_seq_length, special_token_nums, vocab_list):
        self.doc = docs[doc_ix]
        self.vocab_list = vocab_list
        self.doc_ix = doc_ix
        self.docs = docs
        self.special_token_nums = special_token_nums
        self.max_target_length = max_seq_length - special_token_nums  # [CLS], [SEP], [SEP]
        self.instances = []

    def create_sent_pairs(self):
        seg_ix = 0
        segs_length = 0
        segs = []
        tf.logging.debug("********************* create sent sample start **********************")
        tf.logging.debug(self.doc)
        lst_seg_ix = 0
        tf.logging.debug(
            f"doc length: {len(self.doc)}, target length: {self.max_target_length}, total length: {sum([len(i) for i in self.doc])}, length list: {[len(i) for i in self.doc]}")
        while seg_ix <= len(self.doc):
            if segs_length >= self.max_target_length or seg_ix == len(self.doc):
                tf.logging.debug(
                    f"            ************** now segix {seg_ix}, lst segix: {lst_seg_ix} now cum seglength: {segs_length}  ************")
                split_point = 1
                if len(segs) > 1:
                    split_point = random.randint(1, len(segs) - 1)
                is_next = random.random() < 0.5
                token_a, token_b = [], []
                tf.logging.debug(f"split point: {split_point}, segs len: {len(segs)}")
                for ix in range(split_point):
                    token_a.extend(segs[ix])
                if not is_next or len(segs) == 1:
                    rnd_doc = self._random_doc()
                    rnd_doc_split_pt = random.randint(0, len(rnd_doc) - 1)
                    target_b_length = self.max_target_length - len(token_a)
                    while rnd_doc_split_pt < len(rnd_doc) and len(token_b) < target_b_length:
                        token_b.extend(rnd_doc[rnd_doc_split_pt])
                        rnd_doc_split_pt += 1
                    is_next = False
                    seg_ix = seg_ix - len(segs) + split_point
                else:
                    for ix in range(split_point, len(segs)):
                        token_b.extend(segs[ix])
                tf.logging.debug(f"truncate before: {len(token_a)}, {len(token_b)}")
                self._truncate_tokens(token_a, token_b)
                tf.logging.debug(f"truncate after: {len(token_a)}, {len(token_b)}")
                # self.instances.append(
                #     {"length": segs_length, "is_next": is_next, "token_a": token_a, "token_b": token_b, "segs": segs})
                tokens, seg_ids = self._concate_tokens_info(token_a, token_b)
                output_tokens, mask_lm_posisiton, mask_lm_label = self._create_mask(tokens)
                self.instances.append(TraingSample(output_tokens, seg_ids, is_next, mask_lm_posisiton, mask_lm_label))
                tf.logging.debug([len(tokens), tokens])
                tf.logging.debug([len(seg_ids), seg_ids])
                segs_length = 0
                segs = []
                lst_seg_ix = seg_ix
                tf.logging.debug(f"split point: {split_point}")
                tf.logging.debug(
                    f"is next: {is_next}, now seg_ix: {seg_ix}, now token_a_length: {len(token_a)}, now token_b length: {len(token_b)}")

                if seg_ix == len(self.doc):
                    seg_ix += 1
                continue
            segs_length += len(self.doc[seg_ix])
            segs.append(self.doc[seg_ix])
            seg_ix += 1
        random.shuffle(self.instances)
        tf.logging.debug("********************* create sent sample over **********************\n")

    def _concate_tokens_info(self, token_a, token_b):
        tokens, seg_ids = [], []
        tokens.append("[CLS]")
        seg_ids.append(0)
        for t in token_a:
            tokens.append(t)
            seg_ids.append(0)
        tokens.append("[SEP]")
        seg_ids.append(0)
        for t in token_b:
            tokens.append(t)
            seg_ids.append(1)
        tokens.append("[SEP]")
        seg_ids.append(1)
        return tokens, seg_ids

    def _random_doc(self):
        while True:
            rnd_doc_ix = random.randint(0, len(self.docs))
            if rnd_doc_ix == self.doc_ix:
                return self.docs[rnd_doc_ix]

    def _create_mask(self, tokens, masked_lm_prob=0.15, max_prediction_per_seq=20):
        candi_mask_idx = []
        # 由于是全词mask，因此需要先取出候选的mask单位索引
        for ix, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]"]:
                continue
            if len(candi_mask_idx) >= 1 and token.startswith("##"):
                candi_mask_idx[-1].append(ix)
                continue
            candi_mask_idx.append([ix])
        tf.logging.debug(f"candidate idxes {candi_mask_idx}")
        random.shuffle(candi_mask_idx)
        output_tokens = list(tokens)
        num_pred = min(max_prediction_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
        tf.logging.debug(f"max predict num: {num_pred}")
        mask_lms = []
        covered_indexes = set()
        flg = 0
        for idxes in candi_mask_idx:
            if len(mask_lms) >= num_pred:
                break
            # 由于全词掩盖， 对于快要溢出mask词数限制，超出的mask单位直接跳过
            if len(mask_lms) + len(idxes) > num_pred:
                continue
            # 排除mask组里面有一个呗mask过
            is_any_idx_convered = False
            for ix in idxes:
                if ix in covered_indexes:
                    is_any_idx_convered = True
                    break
            if is_any_idx_convered:
                continue

            for ix in idxes:
                covered_indexes.add(ix)
                if random.random() < 0.8:
                    flg += 1
                    mask_token = "[MASK]"
                else:
                    if random.random() < 0.5:
                        mask_token = tokens[ix]
                    else:
                        mask_token = self.vocab_list[random.randint(0, len(self.vocab_list) - 1)]
                output_tokens[ix] = mask_token
                mask_lms.append({"index": ix, "label": tokens[ix]})
        mask_lms = sorted(mask_lms, key=lambda x: x["index"])
        tf.logging.debug(mask_lms)
        tf.logging.debug(f"mask的次序列：{output_tokens}")
        tf.logging.debug(f"mask 数量： {flg}")
        mask_lm_positions = []
        mask_lm_labels = []
        for p in mask_lms:
            mask_lm_positions.append(p["index"])
            mask_lm_labels.append(p["label"])
        return output_tokens, mask_lm_positions, mask_lm_labels

    def _truncate_tokens(self, token_a, token_b):
        while True:
            tnt_length = len(token_b) + len(token_a)
            if tnt_length <= self.max_target_length:
                break
            trunc_token = token_a if len(token_a) > len(token_b) else token_b
            if random.random() < 0.5:
                del trunc_token[0]
            else:
                trunc_token.pop()


class TraingSample:
    def __init__(self, tokens, seg_ids, is_next, mask_lm_position, mask_lm_labels):
        self.tokens = tokens
        self.seg_ids = seg_ids
        self.is_next = is_next
        self.mask_lm_positions = mask_lm_position
        self.mask_lm_labels = mask_lm_labels

    def __str__(self):
        s = ["************** Traing Samples ****************", f"tokens: {len(self.tokens)}--{self.tokens}",
             f"seg_ids: -{len(self.seg_ids)}-{self.seg_ids}", f"is_next: {self.is_next}",
             f"mask_lm_position: -{len(self.mask_lm_positions)}-{self.mask_lm_positions}",
             f"mask_lm_labels: -{len(self.mask_lm_labels)}-{self.mask_lm_labels}"]
        return "\n".join(s)


class WriteCorpusTFDataset:
    def __init__(self, sample_infos, output_files, tokenizer, max_seq_length, max_pred_per_seq):
        self.sample_infos = sample_infos
        self.writers = [tf.python_io.TFRecordWriter(of) for of in output_files]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_pred_per_seq = max_pred_per_seq

    def _format_model_input(self, sample_info):
        input_ids = self.tokenizer.convert_tokens_to_ids(sample_info.tokens)
        input_mask = [1] * len(input_ids)
        seg_ids = sample_info.seg_ids
        assert len(input_ids) <= self.max_seq_length

        for _ in range(self.max_seq_length - len(input_ids)):
            input_ids.append(0)
            input_mask.append(0)
            seg_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(seg_ids) == self.max_seq_length

        mask_lm_position = sample_info.mask_lm_positions
        mask_lm_ids = self.tokenizer.convert_tokens_to_ids(sample_info.mask_lm_labels)
        mask_lm_weights = [1.0] * len(mask_lm_ids)

        for _ in range(self.max_pred_per_seq - len(mask_lm_position)):
            mask_lm_position.append(0)
            mask_lm_ids.append(0)
            mask_lm_weights.append(0.0)

        next_sent_label = int(sample_info.is_next)
        features = collections.OrderedDict()
        features["input_ids"] = self._create_int_feature(input_ids)
        features["input_mask"] = self._create_int_feature(input_mask)
        features["segment_ids"] = self._create_int_feature(seg_ids)
        features["masked_lm_positions"] = self._create_int_feature(mask_lm_position)
        features["masked_lm_ids"] = self._create_int_feature(mask_lm_ids)
        features["masked_lm_weights"] = self._create_float_feature(mask_lm_weights)
        features["next_sent_label"] = self._create_int_feature([next_sent_label])

        self.tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    def writeTF(self):
        write_flg = 0
        for sample_info in self.sample_infos:
            self._format_model_input(sample_info)
            write_idx = write_flg % len(self.writers)
            tf.logging.info(f"write to file {write_idx}")
            write_flg += 1
            writer = self.writers[write_idx]
            writer.write(self.tf_example.SerializeToString())

    def _create_int_feature(self, values):
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return feature

    def _create_float_feature(self, values):
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feature

    def cloas_writers(self):
        for w in self.writers:
            w.close()


class ReadTFrecord:
    def __init__(self, tfrecord_files, batch_size, max_seq_length, max_predictions_per_seq):
        self.tfrecord_files = tfrecord_files
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.batch_size = batch_size
        self.datasets = []

    def _parse(self, record):
        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([self.max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([self.max_predictions_per_seq], tf.float32),
            "next_sent_label":
                tf.FixedLenFeature([1], tf.int64),
        }
        features = tf.parse_single_example(record, name_to_features)
        return features

    def get_batch(self):
        random.shuffle(self.tfrecord_files)
        for tfrecord_file in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            dataset = dataset.map(self._parse).shuffle(100).batch(self.batch_size)
            dataset = dataset.make_one_shot_iterator().get_next()
            self.datasets.append(dataset)
