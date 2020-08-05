"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils
from collections import defaultdict

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

def maybe_place_batch_on_cuda(batch, use_cuda):
    placed_batch = {}
    for name, data in batch.items():
        if name == 'base':
            base_batch = batch['base'][:10]
            labels = batch['base'][10]
            orig_idx = batch['base'][11]
            if use_cuda:
                placed_batch['base'] = [component.cuda() for component in base_batch]
                labels = labels.cuda()
            else:
                placed_batch['base'] = base_batch
        else:
            if use_cuda:
                placed_batch[name] = [component.cuda() for component in data]
            else:
                placed_batch[name] = data
    return placed_batch, labels, orig_idx


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        losses = {}
        # inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        inputs, labels, orig_idx = maybe_place_batch_on_cuda(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output, suppplemental_losses = self.model(inputs)
        main_loss = self.criterion(logits, labels)
        losses['re_loss'] = main_loss.data.item()
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            conv_l2_loss = self.model.conv_l2() * self.opt['conv_l2']
            main_loss += conv_l2_loss
            losses['conv_l2'] = conv_l2_loss.data.item()
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            pooling_l2_loss = self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
            main_loss += pooling_l2_loss
            losses['pooling_l2'] = pooling_l2_loss.data.item()
        if self.opt['link_prediction'] is not None:
            observed_loss = suppplemental_losses['observed'] * (1. - self.opt['link_prediction']['without_observed'])
            predicted_loss = suppplemental_losses['baseline'] * (1. - self.opt['link_prediction']['without_verification'])
            main_loss += (observed_loss + predicted_loss) * self.opt['link_prediction']['lambda']
            observed_loss_value = observed_loss.data.item()
            predicted_loss_value = predicted_loss.data.item()
            losses.update({'kg_observed': observed_loss_value, 'kg_predicted': predicted_loss_value})
        loss_val = main_loss.item()
        # backward
        main_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        # inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        inputs, labels, orig_idx = maybe_place_batch_on_cuda(batch, self.opt['cuda'])
        # orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
