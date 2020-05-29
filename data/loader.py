"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab
from collections import defaultdict
from copy import deepcopy

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, kg_graph=None):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        if kg_graph is not None:
            self.kg_graph = deepcopy(kg_graph)
        else:
            self.kg_graph = defaultdict(lambda: set())

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        self.eval = evaluation
        if not evaluation:
            # indices = list(range(len(data)))
            # random.shuffle(indices)
            # data = [data[i] for i in indices]
            data = self.shuffle_data(data)
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data['base']]
        self.num_examples = len(data)

        # chunk into batches
        # data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        data = self.create_batches(data, batch_size)
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def create_batches(self, data, batch_size):
        batched_data = []
        for batch_start in range(0, len(data['base']), batch_size):
            batch_end = batch_start + batch_size
            batch = defaultdict(list)
            for name, component in data.items():
                batch[name] = component[batch_start:batch_end]
            batched_data.append(batch)
        return batched_data

    def shuffle_data(self, data):
        shuffled_data = defaultdict(list)
        indices = list(range(len(data['base'])))
        random.shuffle(indices)
        for name, component in data.items():
            shuffled_data[name] = component[indices]
        return shuffled_data

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed_data = defaultdict(list)
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            subject_type = 'SUBJ-'+d['subj_type']
            object_type = 'OBJ-'+d['obj_type']
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = [subject_type] * (se-ss+1)
            tokens[os:oe+1] = [object_type] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed_data['base'] += [(tokens, pos, ner, deprel, head, subj_positions,
                                        obj_positions, subj_type, obj_type, relation)]
            # KG Creation
            subject_id = vocab.word2id[subject_type]
            # Offset by 4 because we need it to be zero-indexed instead of 4-index
            #   (['PAD', 'UNK', 'SUBJ-PER', 'SUBJ-ORG', 'OBJ-*'])
            object_id = vocab.word2id[object_type] - 4
            self.kg_graph[(subject_id, relation)].add(object_id)
            self.kg_graph['subjects'].add(subject_id)
            self.kg_graph['objects'].add(object_id)
            self.kg_graph['relations'].add(relation)
            processed_data['kg'] += [(subject_id, relation, object_id)]
        # KG aggregation
        for idx in range(len(processed_data['kg'])):
            instance_subj, instance_rel, instance_obj = processed_data['kg'][idx]
            known_objects = self.kg_graph[(instance_subj, instance_rel)]
            processed_data['kg'][idx] = (instance_subj, instance_rel, known_objects)
        # Convert list to np.array
        for name, component_data in processed_data.items():
            processed_data[name] = np.array(component_data)

        return processed_data

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def ready_base_batch(self, base_batch, batch_size):
        batch = list(zip(*base_batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        tensor_batch = (
        words, masks, pos, ner, deprel, head, subj_positions,
        obj_positions, subj_type, obj_type, rels, orig_idx
        )
        return {'base': tensor_batch, 'sentence_lengths': lens}

    def ready_kg_batch(self, kg_batch, sentence_lengths):
        num_objects = len(self.kg_graph['objects'])
        batch = list(zip(*kg_batch))
        batch, _ = sort_all(batch, sentence_lengths)
        subjects, relations, known_objects = batch
        subjects = torch.LongTensor(subjects)
        relations = torch.LongTensor(relations)
        labels = []
        for sample_objects, relation in zip(known_objects, relations):
            if relation == constant.NO_RELATION_ID:
                sample_known = np.ones(num_objects, dtype=np.float32)
            else:
                sample_known = np.zeros(num_objects, dtype=np.float32)
                sample_known[list(sample_objects)] = 1.
            labels.append(sample_known)
        labels = np.stack(labels, axis=0)
        labels = torch.FloatTensor(labels)
        tensor_batch = (subjects, relations, labels)
        return tensor_batch

    def ready_data_batch(self, batch):
        tensor_batch = defaultdict(list)
        batch_size = len(batch['base'])
        sentence_lengths = [len(x[0]) for x in batch['base']]
        for name, batch_type in batch.items():
            if name == 'base':
                base_components = self.ready_base_batch(base_batch=batch_type, batch_size=batch_size)
                tensor_batch[name] = base_components['base']
            elif name == 'kg':
                tensor_batch[name] = self.ready_kg_batch(kg_batch=batch_type,
                                                         sentence_lengths=sentence_lengths)
        return tensor_batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch = self.ready_data_batch(batch)
        return batch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

