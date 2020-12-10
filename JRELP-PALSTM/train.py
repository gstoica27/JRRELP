"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
from collections import defaultdict
import yaml
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab

def add_kg_model_params(opt, cwd):
    link_prediction_cfg_file = os.path.join(cwd, 'configs', 'kglp_config.yaml')
    with open(link_prediction_cfg_file, 'r') as handle:
        link_prediction_config = yaml.load(handle)
    link_prediction_model = opt['link_prediction']['model']
    params = link_prediction_config[link_prediction_model]
    params['name'] = link_prediction_model
    return params

def create_model_name(opt):
    top_level_name = 'TACRED'
    approach_type = 'PALSTM-JRRELP' if opt['link_prediction'] is not None else 'PALSTM'
    main_name = '{}-{}-{}-{}'.format(
        opt['optim'], opt['lr'], opt['lr_decay'],
        opt['seed']
    )
    if opt['link_prediction'] is not None:
        kglp_task_cfg = opt['link_prediction']
        kglp_task = '{}-{}-{}-{}'.format(
            kglp_task_cfg['lambda'],
            kglp_task_cfg['without_observed'],
            kglp_task_cfg['without_verification'],
            kglp_task_cfg['without_no_relation']
        )
        lp_cfg = opt['link_prediction']['model']
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

cwd = os.getcwd()
on_server = 'Desktop' not in cwd
config_path = os.path.join(cwd, 'configs', f'{"server" if on_server else "local"}_config.yaml')
with open(config_path, 'r') as file:
    opt = yaml.load(file)


#opt = vars(args)
torch.manual_seed(opt['seed'])
np.random.seed(opt['seed'])
random.seed(1234)
if opt['cpu']:
    opt['cuda'] = False
elif opt['cuda']:
    torch.cuda.manual_seed(opt['seed'])

# make opt
# opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']
opt['object_indices'] = vocab.obj_idxs
# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)
test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'], opt, vocab, evaluation=True)

if opt['link_prediction'] is not None:
    opt['link_prediction']['model'] = add_kg_model_params(opt, cwd)
    opt['num_relations'] = len(constant.LABEL_TO_ID)
    opt['num_subjects'] = len(constant.SUBJ_NER_TO_ID) - 2
    opt['num_objects'] = len(constant.OBJ_NER_TO_ID) - 2
    opt['link_prediction']['model']['num_objects'] = opt['num_objects']

model_id = create_model_name(opt)
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_dev_metrics = defaultdict(lambda: -np.inf)
test_metrics_at_best_dev = defaultdict(lambda: -np.inf)

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    current_dev_metrics, _ = scorer.score(dev_batch.gold(), predictions)
    dev_f1 = current_dev_metrics['f1']

    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

    print("Evaluating on test set...")
    predictions = []
    test_loss = 0
    test_preds = []
    for i, batch in enumerate(test_batch):
        preds, probs, loss = model.predict(batch)
        predictions += preds
        test_loss += loss
        test_preds += probs
    predictions = [id2label[p] for p in predictions]
    test_metrics_at_current_dev, _ = scorer.score(test_batch.gold(), predictions)
    test_f1 = test_metrics_at_current_dev['f1']

    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    print("epoch {}: test_loss = {:.6f}, test_f1 = {:.4f}".format(epoch, test_loss, test_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, test_loss, test_f1))

    if best_dev_metrics['f1'] <= current_dev_metrics['f1']:
        best_dev_metrics = current_dev_metrics
        test_metrics_at_best_dev = test_metrics_at_current_dev

    print_str = 'Best Dev Metrics |'
    for name, value in best_dev_metrics.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)
    print_str = 'Test Metrics at Best Dev |'
    for name, value in test_metrics_at_best_dev.items():
        print_str += ' {}: {} |'.format(name, value)
    print(print_str)

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

