"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import yaml
import numpy as np
import json

def generate_param_list(params, cfg_dict, prefix=''):
    param_list = prefix
    for param in params:
        if param_list == '':
            param_list += f'{cfg_dict[param]}'
        else:
            param_list += f'-{cfg_dict[param]}'
    return param_list

def create_model_name(opt):
    top_level_name = 'TACRED'
    approach_type = 'PALSTM-JRRELP' if opt.get('link_prediction', None) is not None else 'PALSTM'
    main_name = '{}-{}-{}-{}'.format(
        opt['optim'], opt['lr'], opt['lr_decay'],
        opt['seed']
    )
    if opt.get('link_prediction', None) is not None:
        kglp_task_cfg = opt.get('link_prediction', None)
        kglp_task = '{}-{}-{}-{}-{}-{}'.format(
            kglp_task_cfg['label_smoothing'],
            kglp_task_cfg['lambda'],
            kglp_task_cfg['freeze_network'],
            kglp_task_cfg['without_observed'],
            kglp_task_cfg['without_verification'],
            kglp_task_cfg['without_no_relation']
        )
        lp_cfg = opt.get('link_prediction', None)['model']
        kglp_name = '{}-{}-{}-{}-{}-{}-{}'.format(
            lp_cfg['input_drop'], lp_cfg['hidden_drop'],
            lp_cfg['feat_drop'], lp_cfg['rel_emb_dim'],
            lp_cfg['use_bias'], lp_cfg['filter_channels'],
            lp_cfg['stride']
        )

        aggregate_name = os.path.join(top_level_name, approach_type, main_name, kglp_task, kglp_name)
    else:
        aggregate_name = os.path.join(top_level_name, approach_type, main_name)
    return aggregate_name

# Copied from: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def create_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def compute_ranks(probs, gold_labels, hits_to_compute=(1, 3, 5, 10, 20, 50)):
    gold_ids = np.array([constant.LABEL_TO_ID[label] for label in gold_labels])
    gold_one_hot = create_one_hot(gold_ids)
    all_probs = np.stack(probs, axis=0)
    all_probs[gold_one_hot == 0] = -np.inf
    all_probs[gold_one_hot == 1] = np.inf
    ranks = np.argsort(all_probs, axis=-1)[:, 0].reshape(-1)
    print(ranks.shape)
    hits = {hits_level: [] for hits_level in hits_to_compute}
    name2ranks = {}
    for hit_level in hits_to_compute:
        valid_preds = np.sum(ranks <= hit_level)
        hits[hit_level] = valid_preds / len(ranks)

    for hit_level in hits:
        name = 'HITs@{}'.format(int(hit_level))
        name2ranks[name] = hits[hit_level]

    mr = np.mean(ranks)
    mrr = np.mean(1. / ranks)
    name2ranks['MRR'] = mrr
    name2ranks['MR'] = mr
    print('RANKS:')
    for name, metric in name2ranks.items():
        print('{}: {}'.format(name, round(metric, 2)))
    return name2ranks


parser = argparse.ArgumentParser()
# parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'{"server" if on_server else "local"}_config.yaml')

def add_kg_model_params(cfg_dict, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'link_prediction_configs.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = cfg_dict.get('link_prediction', None)['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    params['freeze_network'] = cfg_dict.get('link_prediction', None)['freeze_network']
    return params

with open(config_path, 'r') as file:
    cfg_dict = yaml.load(file)

opt = cfg_dict
opt['id'] = create_model_name(opt)
#opt = vars(args)
torch.manual_seed(opt['seed'])
np.random.seed(opt['seed'])
random.seed(1234)
if opt['cpu']:
    opt['cuda'] = False
elif opt['cuda']:
    torch.cuda.manual_seed(opt['seed'])

# load opt
# model_load_dir = opt['save_dir'] + '/' + opt['id']
model_load_dir = '/zfsauton3/home/gis/research/original_palstm/tacred-relation/saved_models/00'
print(model_load_dir)
model_file = os.path.join(model_load_dir, 'best_model.pt')
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['id_load_file'] = None
# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
# Add subject/object indices
opt['object_indices'] = vocab.obj_idxs
# Init Model
model = RelationModel(opt)
model.load(model_file)

# load data
data_file = opt['data_dir'] +f'/test.json'
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
metrics, other_data = scorer.score(batch.gold(), predictions, verbose=True)

compute_ranks(all_probs, gold_labels=batch.gold())

p = metrics['precision']
r = metrics['recall']
f1 = metrics['f1']

wrong_indices = other_data['wrong_indices']
correct_indices = other_data['correct_indices']
wrong_predictions = other_data['wrong_predictions']

raw_data = np.array(batch.raw_data)
wrong_data = raw_data[wrong_indices]
correct_data = raw_data[correct_indices]

wrong_ids = [d['id'] for d in wrong_data]
correct_ids = [d['id'] for d in correct_data]

data_save_dir = os.path.join(os.getcwd(), 'palstm_tacred')
os.makedirs(data_save_dir, exist_ok=True)
print('saving to: {}'.format(data_save_dir))
np.savetxt(os.path.join(data_save_dir, 'correct_ids.txt'), correct_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_ids.txt'), wrong_ids, fmt='%s')
np.savetxt(os.path.join(data_save_dir, 'wrong_predictions.txt'), wrong_predictions, fmt='%s')

id2preds = {d['id']: pred for d, pred in zip(raw_data, predictions)}
json.dump(id2preds, open(os.path.join(data_save_dir, 'id2preds.json'), 'w'))

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

