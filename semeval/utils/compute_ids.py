import os
import json
import numpy as np


# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]


def add_to_dict(dictionary, item):
    if isinstance(item, list):
        for item_ in item:
            if item_ not in dictionary:
                dictionary[item_] = len(dictionary)
    else:
        if item not in dictionary:
            dictionary[item] = len(dictionary)
    return dictionary


if __name__ == '__main__':
    subject_ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    object_ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    pos2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    deprel2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    cwd = '/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/semeval/dataset/semeval/aggcn_semeval'
    files = [os.path.join(cwd, name) for name in ['train_sampled_new.json', 'dev.json', 'test_new.json']]
    all_data = []
    for file in files:
        data = json.load(open(file, 'r'))
        all_data += data

    for d in all_data:
        subj_type = d['subj_type']
        subject_ner2id = add_to_dict(subject_ner2id, subj_type)
        obj_type = d['obj_type']
        object_ner2id = add_to_dict(object_ner2id, obj_type)
        ner2id = add_to_dict(ner2id, d['stanford_ner'])
        deprel2id = add_to_dict(deprel2id, d['stanford_deprel'])
        pos2id = add_to_dict(pos2id, d['stanford_pos'])
    print('Done')
