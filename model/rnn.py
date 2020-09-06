"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers
from model.link_prediction_models import *

def initialize_link_prediction_model(params):
    name = params['name'].lower()
    if name == 'distmult':
        model = DistMult(params)
    elif name == 'conve':
        model = ConvE(params)
    elif name == 'complex':
        model = Complex(params)
    else:
        raise ValueError('Only, {distmult, conve, and complex}  are supported')
    return model

def maybe_place_batch_on_cuda(batch, use_cuda):
    placed_batch = {}
    for name, data in batch.items():
        if name == 'base':
            base_batch = batch['base'][:7]
            labels = batch['base'][7]
            orig_idx = batch['base'][8]
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

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        losses = {}
        inputs, labels, orig_idx = maybe_place_batch_on_cuda(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _, supplemental_losses = self.model(inputs)
        loss = self.criterion(logits, labels)

        if self.opt['link_prediction'] is not None:
            kglp_loss = supplemental_losses['kglp']
            verification_loss = supplemental_losses['verification']
            loss += (kglp_loss + verification_loss) * self.opt['link_prediction']['lambda']
            kglp_loss_value = kglp_loss.data.item()
            verification_loss_value = verification_loss.data.item()
            losses.update({'kglp': kglp_loss_value, 'verification': verification_loss_value})

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        # if self.opt['cuda']:
        #     inputs = [b.cuda() for b in batch[:7]]
        #     labels = batch[7].cuda()
        # else:
        #     inputs = [b for b in batch[:7]]
        #     labels = batch[7]
        inputs, labels, orig_idx = maybe_place_batch_on_cuda(batch, self.opt['cuda'])
        # orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'])
        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        # LP Model
        if opt['link_prediction'] is not None:
            link_prediction_cfg = opt['link_prediction']['model']
            self.rel_emb = nn.Embedding(opt['num_relations'], link_prediction_cfg['rel_emb_dim'])
            self.register_parameter('rel_bias', torch.nn.Parameter(torch.zeros((opt['num_relations']))))
            self.object_indices = torch.from_numpy(np.array(self.object_indices))
            if opt['cuda']:
                self.object_indices = self.object_indices.cuda()
            self.lp_model = initialize_link_prediction_model(link_prediction_cfg)

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.object_indices = opt['object_indices']
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        base_inputs = inputs['base']
        words, masks, pos, ner, deprel, subj_pos, obj_pos = base_inputs # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)
        
        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)
        
        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        if self.opt['link_prediction'] is not None:
            subjects, relations, labels = inputs['kg']
            # object indices to compare against
            object_embs = self.emb(self.object_indices)
            # Obtain embeddings
            subject_embs = self.emb(subjects)
            relation_embs = self.rel_emb(relations)
            # Predict objects with LP model
            kglp_preds = self.lp_model(subject_embs, relation_embs, object_embs)
            verification_preds = self.lp_model(subject_embs, outputs, object_embs)
            # Compute each loss term
            relation_kg_loss = self.kg_model.loss(kglp_preds, labels)
            sentence_kg_loss = self.kg_model.loss(verification_preds, labels)
            if self.opt['link_prediction']['without_no_relation']:
                positive_relations = torch.eq(labels, constant.NO_RELATION_ID).eq(0).type(torch.float32)
                relation_kg_loss = relation_kg_loss * positive_relations
                sentence_kg_loss = sentence_kg_loss * positive_relations
            kglp_loss = relation_kg_loss.mean() * (1 - self.opt['link_prediction']['without_observed'])
            verification_loss = sentence_kg_loss.mean() * (1 - self.opt['link_prediction']['without_verification'])
            supplemental_losses = {'kglp':kglp_loss, 'verification': verification_loss}
            # Remove gradient from flowing to the relation embeddings in the main loss calculation
            logits = torch.mm(final_hidden, self.rel_emb.weight.transpose(1, 0).detach())
            #logits += self.class_bias
        else:
            supplemental_losses = {}
            logits = self.linear(final_hidden)

        return logits, final_hidden, supplemental_losses
    

