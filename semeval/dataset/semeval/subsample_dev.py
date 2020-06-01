import os
import json
import numpy as np
from collections import defaultdict

def load_data(data_file):
    with open(data_file, 'r') as handle:
        return json.load(handle)

def group_by_triple(data):
    triple2data = defaultdict(lambda: [])
    for d in data:
        relation = d['relation']
        if relation == 'Other':
            subj_type = '*'
            obj_type = '*'
        else:
            subj_type, obj_type = relation.split('-')
        relation = d['relation']
        triple = (subj_type, relation, obj_type)
        triple2data[triple].append(d)
    return triple2data

def ascending_triples(triple2data):
    return {k: v for k, v in sorted(triple2data.items(), key=lambda item: len(item[1]))}

def sample_by_triple(data, sample_prop=.1):
    # sorted_data = ascending_triples(data)
    sampled_data = defaultdict()
    retained_data = defaultdict()
    # left_to_sample = int(8000 * sample_prop)
    # triples_left = len(data)
    # per_triple_sample = int(left_to_sample / triples_left)
    for triple, samples in data.items():
        list_samples = np.array(samples)
        sample_idxs = np.arange(len(list_samples))
        np.random.shuffle(sample_idxs)
        sample_num = int(len(samples) * sample_prop)

        chosen_idxs = sample_idxs[:sample_num]
        chosen_samples = list_samples[chosen_idxs]
        for chosen_sample in chosen_samples:
            sampled_data[int(chosen_sample['id'])] = chosen_sample

        retained_idxs = sample_idxs[sample_num:]
        retained_samples = list_samples[retained_idxs]
        for retained_sample in retained_samples:
            retained_data[int(retained_sample['id'])] = retained_sample

    _, sorted_sampled = zip(*list(sorted(sampled_data.items())))
    _, sorted_retained = zip(*list(sorted(retained_data.items())))
    return sorted_retained, sorted_sampled


        # triples_left -= 1
        # samp
if __name__ == '__main__':
    semeval_dir = '/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/semeval/dataset/semeval/aggcn_semeval'
    train_file = os.path.join(semeval_dir, 'train_new.json')
    train_data = load_data(train_file)
    # This gives you exactly an 800 split
    sample_amount = .1006
    triple2data = group_by_triple(train_data)
    train_set, dev_set = sample_by_triple(triple2data, sample_prop=sample_amount)
    train_save_file = os.path.join(semeval_dir, 'train_sampled_new.json')
    json.dump(train_set, open(train_save_file, 'w'))
    dev_save_file = os.path.join(semeval_dir, 'dev.json')
    json.dump(dev_set, open(dev_save_file, 'w'))
