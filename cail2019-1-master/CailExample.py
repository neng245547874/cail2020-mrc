from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math
from answer_verified import *
from io import open
from tqdm import tqdm
from transformers.tokenization_bert import (BasicTokenizer, whitespace_tokenize)

logger = logging.getLogger(__name__)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 is_yes=None,
                 is_no=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.is_yes = is_yes
        self.is_no = is_no

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 unk_mask=None,
                 yes_mask=None,
                 no_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.unk_mask = unk_mask
        self.yes_mask = yes_mask
        self.no_mask = no_mask


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    change_ques_num = 0
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            # paragraph_text =
            # paragraph_text.replace

            doc_tokens = []
            char_to_word_offset = []
            # prev_is_whitespace = True
            # for c in paragraph_text:
            #     if is_whitespace(c):
            #         prev_is_whitespace = True
            #     else:
            #         if prev_is_whitespace:
            #             doc_tokens.append(c)
            #         else:
            #             doc_tokens[-1] += c
            #         prev_is_whitespace = False
            #     char_to_word_offset.append(len(doc_tokens) - 1)

            for c in paragraph_text:
                # if not is_whitespace(c):
                doc_tokens.append(c)
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                # question_text = qa["question"].replace("是多少", "为多少").replace("是谁", "为谁")
                question_text = qa["question"]
                if not question_text:
                    question_text = '是否？？'
                if question_text != qa['question']:#修改
                    change_ques_num += 1
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                is_yes = False
                is_no = False
                if is_training:
                    if version_2_with_negative:
                        if qa['is_impossible'] == 'false':
                            is_impossible = False
                        else:
                            is_impossible = True
                        # is_impossible = qa["is_impossible"]
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     continue
                        # raise ValueError(
                        #     "For training, each question should have exactly 1 answer.")

                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        # actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        actual_text = "".join(doc_tokens[start_position:(end_position + 1)])

                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:

                            if cleaned_answer_text == 'YES':
                                # start_position = max_seq_length+1
                                # end_position = max_seq_length+1
                                is_yes = True
                                orig_answer_text = 'YES'
                                start_position = -1
                                end_position = -1
                            elif cleaned_answer_text == 'NO':
                                is_no = True
                                start_position = -1
                                end_position = -1
                                orig_answer_text = 'NO'
                            else:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    is_yes=is_yes,
                    is_no=is_no)
                examples.append(example)

    logger.info("更改的问题数目为: {}".format(change_ques_num))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    unk_tokens = {}

    convert_token_list = {
        '“': '"', "”": '"', '…': '...', '﹤': '<', '﹥': '>', '‘': "'", '’': "'",
        '﹪': '%', 'Ⅹ': 'x', '―': '-', '—': '-', '﹟': '#', '㈠': '一'
    }
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            # if token in convert_token_list:
            #     token = convert_token_list[token]
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if "[UNK]" in sub_tokens:
                if token in unk_tokens:
                    unk_tokens[token] += 1
                else:
                    unk_tokens[token] = 1

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)


        spans = []

        truncated_query = tokenizer.tokenize(example.question_text, add_special_tokens=False,
                                           max_length=max_query_length)  # 不加分隔符

        sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence

        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):  # 切分片段
            encoded_dict = tokenizer.encode_plus(
                truncated_query,
                span_doc_tokens,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                # doc_stride文档跨度 跨越几个字符构建新文档
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][
                                     : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                            len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                        tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context  # 判断当前位置是否是那一块的中间 找到最好的块

        for span in spans:  # 找到答案片段
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer

            span_is_impossible = example.is_impossible

            start_position = None
            end_position = None

            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    span_is_impossible = True
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            unk_mask, yes_mask, no_mask = [0], [0], [0]
            if is_training and span_is_impossible:
                start_position = max_seq_length
                end_position = max_seq_length
                unk_mask = [1]
            elif is_training and example.is_yes:
                start_position = max_seq_length+1
                end_position = max_seq_length+1
                yes_mask = [1]
            elif is_training and example.is_no:
                start_position = max_seq_length+2
                end_position = max_seq_length+2
                no_mask = [1]

            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("tokens: %s" % " ".join(span["tokens"]))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in span["token_to_orig_map"].items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in span["token_is_max_context"].items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in span["input_ids"]]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in span["attention_mask"]]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in span["token_type_ids"]]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = "".join(span["tokens"][start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    token_is_max_context=span["token_is_max_context"],
                    input_ids=span["input_ids"],
                    input_mask=span["attention_mask"],
                    segment_ids=span["token_type_ids"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    unk_mask=unk_mask,
                    yes_mask=yes_mask,
                    no_mask=no_mask))
            unique_id += 1
    if is_training:
        with open("unk_tokens_clean", "w", encoding="utf-8") as fh:
            for key, value in unk_tokens.items():
                fh.write(key+" " + str(value)+"\n")

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                # feature_null_score = result.unk_logits[0]*2
                # if feature_null_score < score_null:
                #     score_null = feature_null_score
                #     min_null_feature_index = feature_index
                #     null_start_logit = result.unk_logits[0]
                #     null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            # prelim_predictions.append(
            #     _PrelimPrediction(
            #         feature_index=min_null_feature_index,
            #         start_index=512,
            #         end_index=512,
            #         start_logit=null_start_logit,
            #         end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            elif pred.start_index == 512:
                final_text = "unknown"
            elif pred.start_index == 513:
                final_text = "yes"
            else:
                final_text = "no"

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # # # if we didn't include the empty option in the n-best, include it
        # if version_2_with_negative:
        #     if "" not in seen_predictions:
        #         nbest.append(
        #             _NbestPrediction(
        #                 text="",
        #                 start_logit=null_start_logit,
        #                 end_logit=null_end_logit))
        #
        #     # In very rare edge cases we could only have single null prediction.
        #     # So we just create a nonce prediction in this case to avoid failure.
        #     if len(nbest) == 1:
        #         nbest.insert(0,
        #                      _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        #
        # # In very rare edge cases we could have no valid predictions. So we
        # # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text != "unknown":
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = [nbest_json[0]["text"],nbest_json[1]["text"],nbest_json[2]["text"]]

        # if not version_2_with_negative:
        #     all_predictions[example.qas_id] = nbest_json[0]["text"]
        # else:
        #     # predict "" iff the null score - the score of best non-null > threshold
        #     score_diff = score_null - best_non_null_entry.start_logit - (
        #         best_non_null_entry.end_logit)
        #     scores_diff_json[example.qas_id] = score_diff
        #     if score_diff > null_score_diff_threshold:
        #         all_predictions[example.qas_id] = "unknown"
        #     else:
        #         all_predictions[example.qas_id] = best_non_null_entry.text

        if example.question_text.find('是否') >= 0 and (all_predictions[example.qas_id] != "yes" or all_predictions[example.qas_id] != "no"):
            for entry in nbest:
                if entry.text !="yes" or entry.text !="no":
                    continue
                else:
                    all_predictions[example.qas_id] = entry.text
        if example.question_text.find('是否') >= 0 and (
                all_predictions[example.qas_id] != "yes" or all_predictions[example.qas_id] != "no"):
            all_predictions[example.qas_id] = "no"

        if example.question_text == "是否？？":
            all_predictions[example.qas_id] = ["yes"]




        all_nbest_json[example.qas_id] = nbest_json



    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps({'answer':all_predictions,'sp':{}}, indent=4, ensure_ascii=False) + "\n") #修改

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def write_predictions_test(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      verbose_logging, version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.unk_logits[0]*2
                # feature_null_score = result.start_logits[0]+result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            elif pred.start_index == 512:
                final_text = ""
            elif pred.start_index == 513:
                final_text = "YES"
            else:
                final_text = "NO"

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        assert len(nbest) >= 1

        total_scores = []
        # add
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            # if not best_non_null_entry:
            #     if entry.text:
            #         best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        # predict "" iff the null score - the score of best non-null > threshold
        # unk的score和预测最好的span的差值
        # score_diff = score_null - best_non_null_entry.start_logit - (
        #     best_non_null_entry.end_logit)
        # scores_diff_json[example.qas_id] = score_diff
        # if score_diff > null_score_diff_threshold:
        #     all_predictions[example.qas_id] = ""
        # else:
        #     all_predictions[example.qas_id] = best_non_null_entry.text

        all_predictions[example.qas_id] = nbest_json[0]["text"]

    # if version_2_with_negative:
    #         with open(output_null_log_odds_file, "w") as writer:
    #             writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
    #
    # with open(output_nbest_file, "w") as writer:
    #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    # # preds = []
    # # for key, value in all_predictions.items():
    # #     preds.append({'id': key, 'answer': value})
    # #
    # with open(output_prediction_file+'new', 'w') as fh:
    #     json.dump(all_predictions, fh, ensure_ascii=False)

    yes_id = []
    the_insured = {}
    null_id = []
    doc_len = {}
    unk_id = []
    long_answer = {}
    time_id = {}
    occur_time = {}
    repair_r = {}
    insurant_person_id = {}
    insurant_company_id = {}
    for example in all_examples:
        if example.question_text.find('是否') >= 0:
            yes_id.append(example.qas_id)

        if example.question_text.find('吗？') >= 0:
            null_id.append(example.qas_id)

        if find_correct_the_insured(example.question_text, "".join(example.doc_tokens)) != '':
            the_insured[example.qas_id] =\
                find_correct_the_insured(example.question_text, "".join(example.doc_tokens))
        doc_len[example.qas_id] = len(example.doc_tokens)

        # if example.question_text.find('谁') >= 0 or example.question_text.find('何人') >= 0:
        #     who_id.append(example.qas_id)
        if example.question_text in ['被告人判刑情况？',
                                     '被告人有无存在其他犯罪记录？', '哪个法院受理了此案？',
                                     '双方有没有达成一致的调解意见？', '被告人最终判刑情况？',
                                     '被告人是如何归案的？', '本案诉讼费是多少钱？',
                                     '双方有没有达成一致的协调意见？', '本案事实有无证据证实？',
                                     '本案所述事故发生原因是什么？', '事故发生原因是什么？']:
            unk_id.append(example.qas_id)
        if example.question_text.find("案件发生经过是怎样的") >= 0:
            long_answer[example.qas_id] = find_long_answer(all_predictions[example.qas_id], "".join(example.doc_tokens),
                                                           example.question_text)
            print('long_answer')
            print('r', long_answer[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('有效时间是多久') >= 0:
            time_id[example.qas_id] = find_time_span(example.question_text, all_predictions[example.qas_id])

            print('time_id')
            print('r', time_id[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故发生时间是什么时候？') >= 0:
            occur_time[example.qas_id] = repair_time(example.question_text, all_predictions[example.qas_id])
            print('occur_time')
            print('r', occur_time[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故结果如何') >= 0:
            repair_r[example.qas_id] = repair_result("".join(example.doc_tokens),
                                                     example.question_text, all_predictions[example.qas_id])

            print('occur_time')
            print('r', repair_r[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('投保的人是谁') >= 0 or example.question_text.find('投保人是谁') >= 0:
            per = get_insurant_person("".join(example.doc_tokens), example.question_text)
            if per:
                insurant_person_id[example.qas_id] = per
                print('ins_per')
                print('r', insurant_person_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

        if example.question_text.find('向什么公司投保') >= 0:
            cmp = get_insurant_company("".join(example.doc_tokens))
            if cmp:
                insurant_company_id[example.qas_id] = cmp
                print('ins_cmp')
                print('r', insurant_company_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

    preds = []
    for key, value in all_predictions.items():
        if key in insurant_company_id:
            preds.append({'id': key, 'answer': insurant_company_id[key]})
        elif key in insurant_person_id:
            preds.append({'id': key, 'answer': insurant_person_id[key]})
        elif key in long_answer:
            preds.append({'id': key, 'answer': long_answer[key]})
        elif key in time_id:
            preds.append({'id': key, 'answer': time_id[key]})
        elif key in occur_time:
            preds.append({'id': key, 'answer': occur_time[key]})
        elif key in repair_r:
            preds.append({'id': key, 'answer': repair_r[key]})
        elif key in unk_id:
            preds.append({'id': key, 'answer': ''})
        elif key in yes_id:
            if value in ['YES', 'NO', '']:
                preds.append({'id': key, 'answer': value})
            elif value.find('未') >= 0 or value.find('没有') >= 0 or value.find('不是') >= 0 \
                or value.find('无责任') >= 0 or value.find('不归还') >= 0 \
                or value.find('不予认可') >= 0 or value.find('拒不') >= 0 \
                or value.find('无效') >= 0 or value.find('不是') >= 0 \
                or value.find('未尽') >= 0 or value.find('未经') >= 0 \
                or value.find('无异议') >= 0 or value.find('未办理') >= 0\
                or value.find('均未') >= 0:
                preds.append({'id': key, 'answer': "NO"})
            else:
                preds.append({'id': key, 'answer': "YES"})
        elif key in the_insured:
            if value != '' and the_insured[key].find(value) >= 0:
                preds.append({'id': key, 'answer': value})
            else:
                preds.append({'id': key, 'answer': the_insured[key]})

        else:
            preds.append({'id': key, 'answer': value})

    with open(output_prediction_file, 'w') as fh:
        json.dump(preds, fh, ensure_ascii=False)


def find_correct_the_insured(question, passage_all):
    pred_answer = ''
    if question.find('被保险人是谁') >= 0 or (question.find('被保险人是') >= 0 and question.find('被保险人是否') < 0):
        # 还有一种情况，被保险人xxx，但是这种很难匹配因为文章可能出现多次，所以交给模型来预测
        if passage_all.find('被保险人是') >= 0:
            start_index = passage_all.find('被保险人是')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        elif passage_all.find('被保险人为') >= 0:
            start_index = passage_all.find('被保险人为')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        if pred_answer != '' and question.find("被保险人是" + pred_answer) > 0:
            pred_answer = 'YES'

    if question.find('投保人是谁') >= 0:
        start_index = passage_all.find('投保人为')
        for ch in passage_all[start_index + 4:]:
            if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                break
            else:
                pred_answer += ch

    # 如果 pred_answer ==''说明文章中找不到，以模型预测出的结果为准
    return pred_answer


def write_predictions_test_ensemble(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case,
                      verbose_logging, version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    # 将每个样例的不同片段加入到对应的list中， 一个example_index对应若干个unique_id
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    # 每个unique_id的答案
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        # 获得该样本所有片段
        features = example_index_to_features[example_index]

        # 该样本的答案
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        score_yes = 1000000
        min_yes_feature_index = 0  # the paragraph slice with min null score
        yes_start_logit = 0  # the start logit at the slice with min null score
        yes_end_logit = 0  # the end logit at the slice with min null score

        score_no = 1000000
        min_no_feature_index = 0  # the paragraph slice with min null score
        no_start_logit = 0  # the start logit at the slice with min null score
        no_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            # 对于某个片段，计算得分
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.unk_logits[0]*2
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.unk_logits[0]
                    null_end_logit = result.unk_logits[0]

                feature_yes_score = result.yes_logits[0] + result.yes_logits[0]
                if feature_yes_score < score_yes:
                    score_yes = feature_yes_score
                    min_yes_feature_index = feature_index
                    yes_start_logit = result.yes_logits[0]
                    yes_end_logit = result.yes_logits[0]

                feature_no_score = result.no_logits[0] + result.no_logits[0]
                if feature_no_score < score_no:
                    score_no = feature_no_score
                    min_no_feature_index = feature_index
                    no_start_logit = result.no_logits[0]
                    no_end_logit = result.no_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=512,
                    end_index=512,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_yes_feature_index,
                    start_index=513,
                    end_index=513,
                    start_logit=yes_start_logit,
                    end_logit=yes_end_logit))
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_no_feature_index,
                    start_index=514,
                    end_index=514,
                    start_logit=no_start_logit,
                    end_logit=no_end_logit))
        # 排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index < 512:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = "".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = "".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            elif pred.start_index == 512:
                final_text = ""
            elif pred.start_index == 513:
                final_text = "YES"
            else:
                final_text = "NO"

            if final_text in seen_predictions:
                continue
            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # # # if we didn't include the empty option in the n-best, include it
        # if version_2_with_negative:
        #     if "" not in seen_predictions:
        #         nbest.append(
        #             _NbestPrediction(
        #                 text="",
        #                 start_logit=null_start_logit,
        #                 end_logit=null_end_logit))
        #
        #     # In very rare edge cases we could only have single null prediction.
        #     # So we just create a nonce prediction in this case to avoid failure.
        #     if len(nbest) == 1:
        #         nbest.insert(0,
        #                      _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        #
        # # In very rare edge cases we could have no valid predictions. So we
        # # just create a nonce prediction in this case to avoid failure.
        # if not nbest:
        #     nbest.append(
        #         _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        # best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            # if not best_non_null_entry:
            #     if entry.text:
            #         best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json
