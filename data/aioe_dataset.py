# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import random
import math
from torch.nn.utils.rnn import pad_sequence
import copy
from pypinyin import lazy_pinyin

logger = logging.getLogger(__name__)

def gen_bi_context_span_from_sent(tgt_len, pos):
    max_span_length = int( math.ceil( tgt_len * 0.25 ) )
    left_shift = min( random.randint(0, pos), max_span_length )
    right_shift = min( random.randint(1, tgt_len - pos), max_span_length + 1 )
    lhs = max(1, pos - left_shift)  # TODO: change this to 0
    rhs = min(pos + right_shift, tgt_len)
    return lhs, rhs

def gen_bi_context_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    span = gen_bi_context_span_from_sent(tgt_len, pos)
    masked = [0] * tgt_len
    for word_id in range(span[0], span[1]):
        masked[word_id] = 1
    return masked

def gen_suffix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(0, pos):
        masked[word_id] = 1
    return masked

def gen_prefix_mask(tgt_len, pos):
    if tgt_len <= 1:
        return [0] * tgt_len
    masked = [0] * tgt_len
    for word_id in range(pos, tgt_len):
        masked[word_id] = 1
    return masked

def get_pos_with_mask(tgt_mask, cur_len, pad_idx):
    n = 1
    tgt_len = len(tgt_mask)
    row_pos = [0] * tgt_len
    for j in range(cur_len):
        if tgt_mask[j]:
            if j > 0 and tgt_mask[j - 1]:
                row_pos[j] = row_pos[j-1]
            else:
                row_pos[j] = n
                n += 1
        else:
            row_pos[j] = n
            n += 1
    row_pos = [ num+pad_idx for num in row_pos ]
    return row_pos

def gen_bi_context_lr(tgt_len, anchor):
    left = random.randint(0, anchor-1)
    right = random.randint(anchor+1, tgt_len-1)
    return left, right

def get_span(start, end):
    # sample a span from [start, end]
    x = random.randint(start, end)
    y = random.randint(start, end)
    if x > y:
        return y, x
    else:
        return x, y


def sample_typed_char(tgt_token, tgt_dict, symbol2pinyin):
    if type(tgt_token) != str:
        tgt_token = tgt_dict.symbols[tgt_token]
    if symbol2pinyin is None:
        typed_seq_length = random.randint( 1, len(tgt_token) )
        typed_char = tgt_token[ :typed_seq_length ]
    else:
        tgt_token = symbol2pinyin[tgt_token]
        typed_seq_length = random.randint( 1, len(tgt_token) )
        typed_char = tgt_token[ :typed_seq_length ]
    if len(typed_char) == len(tgt_token) and len(typed_char) > 1:
        typed_char = typed_char[:-1]
    return typed_char

def add_word_suggestion_sample(samples, sugst_type, mask_idx, pad_idx, tip_idx, eos_idx, tgt_dict, no_typed_chars, symbol2pinyin=None, target_lang=None, full_target_sentence=False): 
    # seqs: targets from each example
    seqs = [sample["target"] for sample in samples]
    sample_p = [sample["training_sample_p"] for sample in samples]

    batch_size = len(seqs)
    if full_target_sentence == True:
        full_target_sentences = seqs
    seqs = [seq.tolist()[:-1] for seq in seqs]  # remove <eos> token
    target_phrases = []
    tgt_lens = [len(seq) for seq in seqs]
    min_tgt_len = min(tgt_lens)
    # max_tgt_len = input_matrix.size(1)

    seqs_new = []
    anchors = []
    position_ids = []
    segment_ids = []
    target_tokens = []

    len_limit = None
    if target_lang == "zh": 
        len_limit = 2   #TODO: It is needed to change for Chinese.
    else:
        len_limit = 3

    def is_valid_target(word):
        if len(tgt_dict.symbols[word]) <= len_limit:
            if len(tgt_dict.symbols[word]) == 1:
                return False
            #  80% of the time, we reject the short target
            #  20% of the time we accept them
            if random.random() < 0.8:
                return False
        if tgt_dict.symbols[word] == "<unk>":
            return False
        return True
    
    def sample_target_word(seq, low, high, sample_p=None):
        sample_time = 0
        words = [tgt_dict.symbols[seq[j]] for j in range(low, high+1)]
        total_len = sum([len(word) for word in words])
        if sample_p is None:
            sample_p = [len(word)/total_len for word in words]
        else:
            ## process probs
            # 1. larger p means more likely to be generated, need to lower the sample rate
            sample_p = [1/p for p in sample_p]
            # 2. set the prob of <unk> to 0
            for i, word_id in enumerate(seq):
                if tgt_dict.symbols[word_id] == "<unk>":
                    sample_p[i] = 0
            # 3. splice the target split
            sample_p = sample_p[low: high+1]
            total = sum(sample_p)
            if total == 0.0:
                sample_p = [1/len(sample_p) for _ in sample_p]
            else:
                sample_p = [p/total for p in sample_p]
            try:
                assert len(sample_p) == len(words)
            except:
                sample_p = [len(word)/total_len for word in words]

        while sample_time < 3:
            # anchor = random.randint(low, high)
            anchor = np.random.choice(range(low, high+1), p=sample_p)
            if is_valid_target(seq[anchor]):
                break
            sample_time += 1
        return anchor
    
    if min_tgt_len >= 1: # In general, it must satify this.
        for i in range(batch_size):
            cur_tgt_len = tgt_lens[i]
            if sugst_type == "prefix":
                if cur_tgt_len == 1:
                    pos = 0
                    anchor = 0
                    anchors.append(anchor)
                    target_token = seqs[i][anchor]
                    target_tokens.append(target_token)
                    left_context = []
                    right_context = []
                else:
                    anchor = sample_target_word(seqs[i], 1, cur_tgt_len-1, sample_p[i])
                    anchors.append(anchor)

                    left_low, left_high =  get_span(0, anchor-1)
                    left_context = seqs[i][left_low: left_high+1]
                    right_context = []
                    target_token = seqs[i][anchor]
                    target_tokens.append(target_token)

            elif sugst_type == "suffix":
                if cur_tgt_len == 1:
                    pos = 0
                    anchor = 0
                    anchors.append(anchor)
                    target_token = seqs[i][anchor]
                    target_tokens.append(target_token)
                    left_context = []
                    right_context = []
                else:
                    # from the second word to the second last word
                    # if target token is unk then we resample anchor, at most three times
                    anchor = sample_target_word(seqs[i], 0, cur_tgt_len-2, sample_p[i])
                    anchors.append(anchor)

                    target_token = seqs[i][anchor]
                    # target tokens are the target words
                    target_tokens.append(target_token)
                    # left is randomly selected from the first word to the word before the anchor
                    # right is randomly selected from the word after the anchor to the last word
                    left_context = []
                    right_low, right_high = get_span(anchor+1, cur_tgt_len-1)
                    right_context = seqs[i][right_low: right_high+1]

            elif sugst_type == "zero_context":
                if cur_tgt_len == 1:
                    anchor = 0
                    anchors.append(anchor)
                    target_token = seqs[i][anchor]
                    target_tokens.append(target_token)
                    left_context = []
                    right_context = []
                else:
                    # from the second word to the second last word
                    # if target token is unk then we resample anchor, at most three times
                    anchor = sample_target_word(seqs[i], 0, cur_tgt_len-1, sample_p[i])
                    anchors.append(anchor)

                    target_token = seqs[i][anchor]
                    # target tokens are the target words
                    target_tokens.append(target_token)
                    left_context = []
                    right_context = []
            elif sugst_type == "bi_context":                
                # anchor is the position of target word
                if cur_tgt_len < 3:
                    # word </s>
                    anchor = 0
                    anchors.append( anchor )
                    target_token = seqs[i][anchor]
                    target_tokens.append(target_token)
                    left_context = [seqs[i][0]]
                    right_context = [seqs[i][-1]]
                else:
                    # from the second word to the second last word
                    # if target token is unk then we resample anchor, at most three times
                    anchor = sample_target_word(seqs[i], 1, cur_tgt_len-2, sample_p[i])
                    anchors.append(anchor)

                    target_token = seqs[i][anchor]
                    # target tokens are the target words
                    target_tokens.append(target_token)
                    # left is randomly selected from the first word to the word before the anchor
                    # right is randomly selected from the word after the anchor to the last word
                    left_low, left_high =  get_span(0, anchor-1)
                    right_low, right_high = get_span(anchor+1, cur_tgt_len-1) 

                    left_context = seqs[i][ left_low : left_high+1 ]
                    right_context = seqs[i][ right_low : right_high+1 ]
            else:
                raise NotImplementedError
            

            typed_char = sample_typed_char(target_token, tgt_dict, symbol2pinyin)
            if no_typed_chars == False:
                # left context + tip + typed_char + tip + mask + right context
                seqs_new.append(left_context + [tip_idx] + [tgt_dict.index(c) for c in typed_char ] + [tip_idx] + [mask_idx] + right_context)
            else:
                seqs_new.append(left_context + [mask_idx] + right_context)
            position_ids.append(list(range(2, 2+len(seqs_new[-1]))))
            segment_ids.append( [1 for _ in seqs_new[-1]] )
            # target phrases are the left context + target word + right context + eos
            # target_phrases.append( seqs[i][left_high+1:right_low] + [eos_idx] )
            # 0507 update: use full target sequence as target phrase
            target_phrases.append( seqs[i] + [eos_idx] )

    if full_target_sentence == True:
        return seqs_new, position_ids, segment_ids, target_tokens, full_target_sentences 
    else:
        return seqs_new, position_ids, segment_ids, target_tokens, target_phrases


def convert_list_to_tensor(lists, padding_value):
    if type(lists[0]) == list:
        lists = [torch.tensor(lst, dtype=torch.int64) for lst in lists]
    lists = pad_sequence(lists, batch_first=True, padding_value=padding_value)  
    return lists

def collate(
    samples,
    pad_idx,
    eos_idx,
    mask_idx,
    tip_idx,
    suggestion_type,
    tgt_dict,
    split,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    types=None,
    suggestions=None,
    decoder_word_prediction=None,
    no_typed_chars=None,
    symbol2pinyin=None,
    target_lang=None,
    full_target_sentences=None
):
    """
    Note by galaxycchen:
    The input sample is a list of "__getitem__" output.
    Include three keys: id, source, target
    source, target have "</s>" in the end
    """
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])

    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if suggestion_type == "full":
        suggestion_type = random.choice(["bi_context", "prefix", "suffix", "zero_context"])
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)

        if split == "train":
            context_tokens, context_postion_ids, context_segment_ids, target_tokens, target_phrases = add_word_suggestion_sample(samples, suggestion_type, mask_idx, pad_idx, tip_idx, eos_idx, tgt_dict, no_typed_chars, symbol2pinyin, target_lang, full_target_sentences)

            src_tokens = [s["source"].tolist() for s in samples]
            src_position_ids = [ list(range(2, 2+len(s))) for s in src_tokens]
            src_segment_ids = [ [0 for _ in range(len(s))] for s in src_tokens]

            for i in range(len(src_tokens)):
                src_tokens[i] = src_tokens[i] + context_tokens[i]
                src_position_ids[i] = src_position_ids[i] + context_postion_ids[i]
                src_segment_ids[i] = src_segment_ids[i] + context_segment_ids[i]

            src_tokens = convert_list_to_tensor(src_tokens, pad_idx)
            src_tokens = src_tokens.index_select(0, sort_order)
            src_position_ids = convert_list_to_tensor(src_position_ids, 1)
            src_position_ids = src_position_ids.index_select(0, sort_order)
            src_segment_ids = convert_list_to_tensor(src_segment_ids, 1)
            src_segment_ids = src_segment_ids.index_select(0, sort_order)
            target_tokens = torch.tensor(target_tokens, dtype=torch.int64)
            target_tokens = target_tokens.index_select(0, sort_order)

            assert src_tokens.size(1) == src_position_ids.size(1) == src_segment_ids.size(1)

            # next, we need to get the prev_output_tokens for the decoder
            prev_output_tokens = data_utils.collate_tokens(
                [torch.tensor(phrase) for phrase in target_phrases],
                pad_idx,
                eos_idx,
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            
            target_phrases = convert_list_to_tensor(target_phrases, pad_idx)
            target_phrases = target_phrases.index_select(0, sort_order)

            assert prev_output_tokens.size(1) == target_phrases.size(1)

            tgt_lengths = target_phrases.ne(pad_idx).long().sum(dim=-1)
            ntokens = tgt_lengths.sum().item() + src_tokens.size(0)                                                                          
        elif split == "valid" or split == "test":
            assert types != None
            context_tokens, context_postion_ids, context_segment_ids, target_tokens = [], [], [], []
            
            for s in samples:
                sample_id = s["id"]
                typed_char = types[sample_id]
                # find <mask> token
                mask_position_idx = s["target"].tolist().index( tgt_dict.index("<mask>") )
                if no_typed_chars == False:
                    context_tokens.append( s["target"].tolist()[:mask_position_idx] + [tip_idx] + [tgt_dict.index(char) for char in typed_char] + [tip_idx] + [mask_idx] + s["target"].tolist()[mask_position_idx+1:] )
                else:
                    context_tokens.append( s["target"].tolist()[:mask_position_idx] + [mask_idx] + s["target"].tolist()[mask_position_idx+1:] )
                context_postion_ids.append( list(range(2, 2+len(context_tokens[-1]))) )
                context_segment_ids.append( [1 for _ in context_tokens[-1]] )
                target_tokens.append( tgt_dict.index(suggestions[sample_id]) )

            src_tokens = [s["source"].tolist() for s in samples]
            src_position_ids = [ list(range(2, 2+len(s))) for s in src_tokens]
            src_segment_ids = [ [0 for _ in range(len(s))] for s in src_tokens]

            for i in range(len(src_tokens)):
                src_tokens[i] = src_tokens[i] + context_tokens[i]
                src_position_ids[i] = src_position_ids[i] + context_postion_ids[i]
                src_segment_ids[i] = src_segment_ids[i] + context_segment_ids[i]

            src_tokens = convert_list_to_tensor(src_tokens, pad_idx)
            src_tokens = src_tokens.index_select(0, sort_order)
            src_position_ids = convert_list_to_tensor(src_position_ids, 1)
            src_position_ids = src_position_ids.index_select(0, sort_order)
            src_segment_ids = convert_list_to_tensor(src_segment_ids, 1)
            src_segment_ids = src_segment_ids.index_select(0, sort_order)
            target_tokens = torch.tensor(target_tokens, dtype=torch.int64)
            target_tokens = target_tokens.index_select(0, sort_order)

            assert src_tokens.size(1) == src_position_ids.size(1) == src_segment_ids.size(1)                                      

            # For valid and test dataset, there is no prev_output_tokens and target_phrases
            prev_output_tokens = torch.full(size=(src_tokens.size(0),1), fill_value=eos_idx)    #* prev_output_tokens are not used 
            target_phrases = torch.full(size=(src_tokens.size(0),1), fill_value=eos_idx)    #* target_phrases are not used
            ntokens = src_tokens.size(0)        
    else:
        raise Exception("No target in samples")

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_position_ids": src_position_ids,
            "src_segment_ids": src_segment_ids,
            "prev_output_tokens": prev_output_tokens,
        },
        "target": target_phrases,
        "target_tokens": target_tokens,
        "suggestion_type": suggestion_type,
    }

    return batch


class AIOEDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        suggestion_type=None,
        split=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        types=None,
        suggestions=None,
        decoder_word_prediction=None,
        no_typed_chars=None,
        symbol2pinyin=None,
        target_lang=None,
        full_target_sentences=None,
        training_sample_p=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.suggestion_type = suggestion_type
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.mask_idx = self.tgt_dict.index("<mask>")
        self.tip_idx = self.tgt_dict.index("<tip>")
        self.split = split
        self.types = types
        self.suggestions = suggestions
        self.decoder_word_prediction = decoder_word_prediction
        self.no_typed_chars = no_typed_chars
        self.symbol2pinyin = symbol2pinyin
        self.target_lang = target_lang
        self.full_target_sentences = full_target_sentences

        self.training_sample_p = training_sample_p
        if self.training_sample_p is not None:
            logger.info("Loading sample p from {}".format(self.training_sample_p))
            training_sample_p_idx = open(self.training_sample_p, "r").readlines()
            training_sample_p_idx = [int(loc) for loc in training_sample_p_idx]
            training_sample_p_file = training_sample_p.strip(".idx")

            self.training_sample_p_idx = training_sample_p_idx
            self.training_sample_p_file = training_sample_p_file

    def get_batch_shapes(self):
        return self.buckets
    
    def get_sample_p(self, index):
        position = self.training_sample_p_idx[index]
        with open(self.training_sample_p_file) as f:
            f.seek(position)
            sample_p = f.readline().strip().split()
        sample_p = [float(p) for p in sample_p]
        return sample_p

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        if self.training_sample_p is not None:
            example["training_sample_p"] = self.get_sample_p(index)
        else:
            example["training_sample_p"] = None
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            mask_idx=self.mask_idx,
            tip_idx=self.tip_idx,
            suggestion_type=self.suggestion_type,
            tgt_dict=self.tgt_dict,
            split=self.split,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            types=self.types,
            suggestions=self.suggestions,
            decoder_word_prediction=self.decoder_word_prediction,
            no_typed_chars=self.no_typed_chars,
            symbol2pinyin=self.symbol2pinyin,
            target_lang=self.target_lang,
            full_target_sentences=self.full_target_sentences
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
