# Based off of https://github.com/TimDettmers/ConvE/blob/master/model.py
# The vast majority of code is equivalent to that file, with the exception
# of a few compatibility (with JRRELP) changes.
# This file contains the ConvE KGLP model. More can be added so long as the input/output process matches ConvE
from operator import mul
from functools import reduce
import torch
from torch.nn import functional as F, Parameter

__all__ = ['ConvE']

class ConvE(torch.nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        self.inp_drop = torch.nn.Dropout(args['input_drop'])
        self.hidden_drop = torch.nn.Dropout(args['hidden_drop'])
        self.feature_map_drop = torch.nn.Dropout2d(args['feat_drop'])
        self.loss = torch.nn.BCELoss(reduce=False)
        self.ent_emb_dim1 = args['ent_emb_shape1']
        self.ent_emb_dim2 = args['ent_emb_dim'] // self.ent_emb_dim1
        self.rel_emb_dim1 = args['rel_emb_shape1']
        self.rel_emb_dim2 = args['rel_emb_dim'] // self.rel_emb_dim1

        self.kernel_size = eval(args['kernel_size'])
        self.filter_channels = args['filter_channels']
        self.stride = args['stride']
        self.padding = args['padding']
        self.ent_emb_dim = args['ent_emb_dim']
        self.rel_emb_dim = args['rel_emb_dim']
        self.conv1 = torch.nn.Conv2d(1, self.filter_channels, self.kernel_size,
                                     self.stride, self.padding, bias=args['use_bias'])
        output_width = (self.ent_emb_dim1 + self.rel_emb_dim1) - self.kernel_size[0] + 1
        output_height = self.ent_emb_dim2 - self.kernel_size[1] + 1
        output_size = output_height * output_width * self.filter_channels
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args['ent_emb_dim'])
        # offset b/c we don't include subjects in calculation
        self.register_parameter('b', Parameter(torch.zeros((args['num_objects']))))
        self.fc = torch.nn.Linear(output_size,args['ent_emb_dim'])

    def forward(self, e1, rel, e2s):
        e1_embedded = e1.view(-1, 1, self.ent_emb_dim1, self.ent_emb_dim2)
        # Assume relation is already encoded form RE model
        rel_embedded = rel.view(-1, 1, self.rel_emb_dim1, self.rel_emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x = torch.mm(x, e2s.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

    def load_model(self, model_path, freeze_network=False):
        state_dict = torch.load(model_path)
        relevant_state_dict = {}
        for (k,v) in state_dict.items():
            if k != 'b' and 'emb' not in k:
                relevant_state_dict[k] = v
            elif k == 'b' and reduce(mul, v.size()) == 19:
                relevant_state_dict[k] = v[2:]

        self.load_state_dict(relevant_state_dict)
        # Only non-entity model parameters can be trained. This forces learned sentence encoding to
        # align with pre-trained relation embeddings, which are already used to evaluate the RE model's
        # loss through the decoder matching layer
        if freeze_network:
            for parameter in self.parameters():
                parameter.requires_grad = False