import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import MIXED_OPS
import dgl.function as fn


class OpModule(nn.Module):

    def __init__(self, args, operation_name):
        super(OpModule, self).__init__()
        self.args = args
        self._feature_dim = args.feature_dim
        args = {'feature_dim': self._feature_dim}
        self.op = MIXED_OPS[operation_name](args)
        self.linear = nn.Linear(self._feature_dim, self._feature_dim, bias=True)
        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()

    def forward(self, g, h, h_in):
        h = self.op(g, h, h_in)
        h = self.linear(h)
        if self.args.op_norm:
            h = self.batchnorm_h(h)
        h = self.activate(h)
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


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class mean_aggre(nn.Module):
    def __init__(self, feature_dim):
        super(mean_aggre, self).__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, block, src_emb):
        src_emb = F.relu(self.linear(src_emb))
        block.edata['msg_e'] = src_emb
        msg = fn.copy_edge('msg_e', 'm')
        block.update_all(msg, reduce_mean)
        h_node = block.dstdata['h']
        return h_node


class Network(nn.Module):

    def __init__(self, device, genotype, number_of_nodes, num_classes, num_rels, layers, zero_nodes, nodes, feature_dim,
                 init_fea_dim, num_base_r, criterion, args):
        super(Network, self).__init__()
        self._device = device
        self._layers = layers
        self._in_dim_n = number_of_nodes
        self._in_dim_e = num_rels
        self._feature_dim = feature_dim
        self._init_fea_dim = init_fea_dim
        self._num_base_r = num_base_r
        self._num_classes = num_classes
        self._criterion = criterion
        self._nb_zero_nodes = zero_nodes
        self._nb_first_nodes = nodes
        self._nb_last_nodes = nodes
        self._nb_zero_edges = self._nb_zero_nodes
        # !!!: this can be adjust, 1, 2...
        self._nb_first_edges = sum(self._nb_zero_nodes + i for i in range(self._nb_first_nodes))
        self._nb_middle_edges = self._nb_first_nodes
        self._nb_last_edges = sum(self._nb_first_nodes + i for i in range(self._nb_last_nodes))
        self.embedding_h = nn.Embedding(self._in_dim_n, self._init_fea_dim)  # node feat is an integer
        self.embedding_e = nn.Embedding(self._num_base_r, self._init_fea_dim)
        self.rel_wt = self.get_param([self._in_dim_e, self._num_base_r])
        self.rel_num = torch.arange(self._num_base_r, device=self._device)
        self.embedding_h_init = nn.Linear(self._init_fea_dim, self._feature_dim, bias=False)
        self.embedding_e_init = nn.Linear(self._init_fea_dim, self._feature_dim, bias=False)

        # print("param size = %fMB", count_parameters_in_MB(self))

        # if type(self._genotype) == list:
        # 	genotypes = self._genotype
        # else:
        # 	genotypes = [self._genotype for i in range(self._layers)]
        # self.cells = nn.ModuleList([Cell(args, genotype[i]) for i in range(self._layers)])

        self.cells = nn.ModuleList([Cell(args, genotype[i]) for i in range(self._layers)])
        outdim = self._feature_dim
        self.classifier = MLPClassifier(outdim, self._num_classes)
        self.mean_aggre = mean_aggre(self._feature_dim)
        # self.concat_weights = nn.Linear((nb_first_nodes + nb_last_nodes) * feature_dim, feature_dim)
        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()

    def _forward(self, trip_index, block):
        src_b_ls = []
        dst_b_ls = []
        eid_b_ls = []
        for j in range(len(block)):
            indices = block[j].edata[dgl.EID]
            eid_b = block[j].edata[dgl.ETYPE]
            eid_b_ls.append(eid_b)
            # trip_index = eid_g, src_g, dst_g
            src_dst_block = torch.index_select(trip_index, dim=0, index=indices)
            src_b = src_dst_block[:, 1]
            src_b_ls.append(src_b)
            dst_b = src_dst_block[:, 2]
            dst_b_ls.append(dst_b)

        for i, cell in enumerate(self.cells):
            if i == 0:
                src_embed = self.embedding_h_init(self.embedding_h(src_b_ls[i]))

            ## @Amanda: we can add a linear layer with num_rel* num_basi
            # edges_embed
            edges_embed = self.embedding_e_init(torch.mm(self.rel_wt[eid_b_ls[i], :], self.embedding_e(self.rel_num)))
            node_embed = cell(block[i], src_embed, edges_embed)
            if i < len(src_b_ls) - 1:
                src_b_tmp = torch.empty(src_b_ls[i + 1].shape, dtype=src_b_ls[i + 1].dtype)
                for m in range(block[i].dstdata[dgl.NID].shape[0]):
                    src_b_tmp[src_b_ls[i + 1] == block[i].dstdata[dgl.NID][m]] = m
                src_embed = node_embed[src_b_tmp, :]

        h = self.batchnorm_h(node_embed)
        h = self.activate(h)
        # print('final h', h.shape)
        return h

    def forward(self, trip_index, g):
        h = self._forward(trip_index, g)
        logits = self.classifier(h)
        return logits

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def _loss(self, trip_index, g, labels, idx):
        logits = self.forward(trip_index, g, )
        loss = self._criterion(logits, labels[idx])
        return loss
