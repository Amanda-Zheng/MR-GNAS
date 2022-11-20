import time

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations_lp import MIXED_OPS, MIXED_OPS_sf
import dgl.function as fn


class OpModule(nn.Module):

    def __init__(self, args, operation_name):
        super(OpModule, self).__init__()
        self.args = args
        self._feature_dim = args.feature_dim
        args = {'feature_dim': self._feature_dim, 'drop_aggr': args.drop_aggr}
        self.op = MIXED_OPS[operation_name](args)
        self.op_name = operation_name
        #self.linear = nn.Linear(self._feature_dim, self._feature_dim, bias=True)
        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()
        self.drop_op = self.args.drop_op

    def forward(self, g, h, h_in):
        h = self.op(g, h, h_in)
        #h = self.linear(h)
        #if self.args.op_norm:
        if self.op_name != 'pre_mult' and 'pre_add' and 'pre_sub':
            h = self.batchnorm_h(h)
            h = self.activate(h)
            F.dropout(h, self.drop_op, training=self.training)
        return h


class Cell(nn.Module):

    def __init__(self, args, genotype):
        super(Cell, self).__init__()
        self.args = args
        self._genotype = genotype
        self._nb_nodes = len(set([edge[1] for edge in genotype.alpha_cell]))
        self._feature_dim = args.feature_dim
        self._concat_node = list(range(1, 1 + self._nb_nodes)) if genotype.concat_node is None else genotype.concat_node
        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()
        self._compile()

    def _compile(self):
        nb_nodes = self._nb_nodes
        self._ops = nn.ModuleList([nn.ModuleList([nn.ModuleList() for i in range(n)]) for n in range(1, 1 + nb_nodes)])
        for (op_name, center_node, pre_node) in self._genotype.alpha_cell:
            center_node -= 1
            self._ops[center_node][pre_node].append(OpModule(self.args, op_name))
        self.concat = nn.Linear(len(self._concat_node) * self._feature_dim, self._feature_dim)

    def forward(self, g, src_emb, hr):
        zero_out = self._ops[0][0][0](g, src_emb, hr)
        states = [src_emb, zero_out]
        for n in range(1, self._nb_nodes):
            hs = []
            for i in range(n + 1):
                if len(self._ops[n][i]) > 0:
                    _h = self._ops[n][i][0](g, states[i], zero_out)
                    hs.append(_h)
            states.append(sum(hs))

        states = [states[idx] for idx in self._concat_node]
        h = self.concat(torch.cat(states, dim=1))
        h = self.batchnorm_h(h)
        h = self.activate(h)
        return h


class Network(nn.Module):

    def __init__(self, device, genotype, number_of_nodes, num_rels, feature_dim, init_fea_dim, num_base_r, criterion, dropout_cell, args):
        super(Network, self).__init__()
        self._device = device
        self._num_ent = number_of_nodes
        self._num_rel = num_rels * 2 + 1
        self._feature_dim = feature_dim
        self.num_base_r = num_base_r
        self.init_fea_dim = init_fea_dim
        self.criterion = criterion

        self.embedding_h = nn.Embedding(self._num_ent, self.init_fea_dim)  # node feat is an integer
        # self.embedding_e = nn.Embedding(self.num_base_r, self.init_fea_dim)
        self.embedding_e = nn.Embedding(self.num_base_r, self._feature_dim)

        # here is for the v2 version
        self.linear_e = nn.Linear(self.init_fea_dim, self._feature_dim)

        # self.linear_e_init = nn.Linear(self.init_fea_dim, self._feature_dim)
        # self.linear_r_init = nn.Linear(self.init_fea_dim, self._feature_dim)
        # self.linear_r_final = nn.Linear(self._feature_dim, self._feature_dim)

        self.idx_ent = torch.arange(self._num_ent, device=self._device)
        self.idx_rel = torch.arange(self.num_base_r, device=self._device)
        # print("param size = %fMB", count_parameters_in_MB(self)

        #self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        #self.activate = nn.ReLU()

        self.rel_wt = self.get_param([self._num_rel, self.num_base_r])

        self.cells = nn.ModuleList([Cell(args, genotype[i]) for i in range(len(genotype))])

        self.score_func = MIXED_OPS_sf[genotype[-1].score_func]({'gamma': args.gamma,
                                                                 'embed_dim': args.embed_dim,
                                                                 'conve_hid_drop': args.conve_hid_drop,
                                                                 'feat_drop': args.feat_drop,
                                                                 'num_filt': args.num_filt,
                                                                 'ker_sz':args.ker_sz,
                                                                 'k_w': args.k_w,
                                                                 'k_h': args.k_h})

        self.w_rel = self.get_param([self._feature_dim, self._feature_dim])
        self._dropout = dropout_cell

    def _forward_lp(self, g, subj, rel):
        all_ent_emb = self.linear_e(self.embedding_h(self.idx_ent))
        rel_embed = torch.mm(self.rel_wt, self.embedding_e(self.idx_rel))
        src_id, _, _ = g.edges(form='all')
        src_id_final = torch.cat((src_id, g.nodes()), dim=0)
        edge_self = (torch.ones(g.nodes().shape) * (self._num_rel - 1)).long().to(self._device)
        edge_type_final = torch.cat((g.edata['e_type'].long(), edge_self), dim=0)
        for i, cell in enumerate(self.cells):
            all_ent_emb = cell(g, all_ent_emb[src_id_final], rel_embed[edge_type_final])
            all_ent_emb = F.dropout(all_ent_emb, self._dropout, training=self.training)
            rel_embed = torch.matmul(rel_embed, self.w_rel)

        score_f = self.score_func(all_ent_emb, all_ent_emb[subj], rel_embed[rel])

        return score_f

    def forward(self, g, subj, rel):
        score_f = self._forward_lp(g, subj, rel)
        return score_f

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def _loss(self, g, subj, rel, label):
        pred = self.forward(g, subj, rel)
        return self.criterion(pred, label)
