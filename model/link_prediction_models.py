# Copied from https://github.com/TimDettmers/ConvE/blob/master/model.py

import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['Complex', 'DistMult', 'ConvE']

class Complex(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()
        self.inp_drop = torch.nn.Dropout(args['input_drop'])
        self.is_pretrained = False
        self.embedding_dim = args['embedding_dim']

    def forward(self, e1, rel, e2):
        # [B, T, E]
        e1_embedded = e1
        rel_embedded = rel
        e2_embedded = e2
        # e1_embedded = e1_embedded.squeeze()
        # rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)
        # ([B, T, E] * [B, 1, E]) --> [B, T, E], [B, T, E] x [B, E, 1] --> [B, T, 1]
        logits = torch.bmm(e1_embedded*rel_embedded, e2_embedded.transpose(2,1))
        return logits

class ConvE(torch.nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        # self.emb_e = torch.nn.Embedding(args['num_entities'],
        #                                 args['ent_emb_dim'])
        # self.emb_rel = torch.nn.Embedding(args['num_relations'],
        #                                   args['embedding_dim'])

        self.inp_drop = torch.nn.Dropout(args['input_drop'])
        self.hidden_drop = torch.nn.Dropout(args['hidden_drop'])
        self.feature_map_drop = torch.nn.Dropout2d(args['feat_drop'])
        self.loss = torch.nn.BCELoss()
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
        # load model if exists
        # if args['load_path'] is not None:
        #     self.load_model(args['load_path'])
        #     self.is_pretrained = True
        # else:
        #     self.is_pretrained = False

    def forward(self, e1, rel, e2s):
        #print('Are cuda? | e1: {} | emb_e: {} | rel: {}'.format(e1.is_cuda, self.emb_e.weight.is_cuda, rel.is_cuda))
        # e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e1_embedded = e1.view(-1, 1, self.ent_emb_dim1, self.ent_emb_dim2)
        # rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)
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

    # def load_model(self, model_path):
    #     state_dict = torch.load(model_path)
    #     # relevant_state_dict = dict([(k,v) for k,v in state_dict.items() if 'emb' not in k and k != 'b'])
    #     self.load_state_dict(state_dict)
    #     # Only non-entity model parameters can be trained. This forces learned sentence encoding to
    #     # align with pre-trained relation embeddings, which are already used to evaluate the RE model's
    #     # loss through the decoder matching layer
    #     if self.opt['freeze_embeddings']:
    #         self.emb_e.weight.requires_grad = False
    #         self.emb_rel.weight.requires_grad = False