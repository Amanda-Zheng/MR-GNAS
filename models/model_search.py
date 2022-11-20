import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configs.genotypes import Genotype
from models.operations import PRE_OPS, FIRST_OPS, MIDDLE_OPS, LAST_OPS
from models.cell import Cell
from functools import partial
import dgl.function as fn

from utils.utils import count_parameters_in_MB


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

    def __init__(self, device, number_of_nodes, num_classes, num_rels, layers, zero_nodes, nodes, feature_dim,
                 init_fea_dim, num_base_r,
                 dropout=0.0):
        super(Network, self).__init__()
        self._device = device
        self._layers = layers
        self._in_dim_n = number_of_nodes
        self._in_dim_e = num_rels
        self._feature_dim = feature_dim
        self._num_base_r = num_base_r
        self._init_fea_dim = init_fea_dim
        self._num_classes = num_classes
        self._criterion = nn.CrossEntropyLoss()
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
        # print("param size = %fMB", count_parameters_in_MB(self)

        self.rel_wt = self.get_param([self._in_dim_e, self._num_base_r])
        self.rel_num = torch.arange(self._num_base_r, device=self._device)
        self.embedding_h_init = nn.Linear(self._init_fea_dim, self._feature_dim, bias=False)
        self.embedding_e_init = nn.Linear(self._init_fea_dim, self._feature_dim, bias=False)

        self.cells = nn.ModuleList(
            [Cell(self._nb_zero_nodes, self._nb_first_nodes, self._nb_last_nodes, self._feature_dim)
             for i in range(self._layers)])
        self._initialize_alphas()
        outdim = self._feature_dim
        self.classifier = MLPClassifier(outdim, self._num_classes)
        self.mean_aggre = mean_aggre(self._feature_dim)
        # self.concat_weights = nn.Linear((nb_first_nodes + nb_last_nodes) * feature_dim, feature_dim)
        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()
        self._dropout = dropout

    def new(self):
        model_new = Network(self._device, self._in_dim_n, self._num_classes, self._layers, self._nb_zero_nodes,
                            self._nb_first_nodes, self._feature_dim).to(self._device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)

    def arch_parameters(self):
        return self._arch_parameters

    def _initialize_alphas(self):
        nb_layers = self._layers
        nb_zero_edges = self._nb_zero_edges
        nb_first_edges = self._nb_first_edges
        nb_middle_edges = self._nb_middle_edges
        nb_last_edges = self._nb_last_edges
        nb_zero_ops = len(PRE_OPS)
        nb_first_ops = len(FIRST_OPS)
        nb_middle_ops = len(MIDDLE_OPS)
        nb_last_ops = len(LAST_OPS)

        self.alphas_zero_cell = Variable(1e-3 * torch.randn(nb_zero_edges * nb_layers, nb_zero_ops).to(self._device),
                                         requires_grad=True)
        self.alphas_first_cell = Variable(1e-3 * torch.randn(nb_first_edges * nb_layers, nb_first_ops).to(self._device),
                                          requires_grad=True)
        self.alphas_middle_cell = Variable(
            1e-3 * torch.randn(nb_middle_edges * nb_layers, nb_middle_ops).to(self._device),
            requires_grad=True)
        self.alphas_last_cell = Variable(1e-3 * torch.randn(nb_last_edges * nb_layers, nb_last_ops).to(self._device),
                                         requires_grad=True)

        self._arch_parameters = [
            self.alphas_zero_cell,
            self.alphas_first_cell,
            self.alphas_middle_cell,
            self.alphas_last_cell,
        ]

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
            W_zero, W_first, W_middle, W_last = self.show_weights(i)
            node_embed = cell(block[i], src_embed, edges_embed, W_zero, W_first, W_middle, W_last)
            if i < len(src_b_ls) - 1:
                src_b_tmp = torch.empty(src_b_ls[i + 1].shape, dtype=src_b_ls[i + 1].dtype)
                for m in range(block[i].dstdata[dgl.NID].shape[0]):
                    src_b_tmp[src_b_ls[i + 1] == block[i].dstdata[dgl.NID][m]] = m
                src_embed = node_embed[src_b_tmp, :]

        h = self.batchnorm_h(node_embed)
        h = self.activate(h)
        h = F.dropout(h, self._dropout, training=self.training)
        # print('final h', h.shape)
        return h

    def forward(self, trip_index, g):
        h = self._forward(trip_index, g)
        logits = self.classifier(h)
        return logits

    def _loss(self, trip_index, g, labels, idx):
        logits = self.forward(trip_index, g, )
        loss = self._criterion(logits, labels[idx])
        return loss

    def normalize_weights(self, W_zero, W_first, W_middle, W_last):
        W_zero = F.softmax(W_zero, dim=1)
        W_first = F.softmax(W_first, dim=1)
        W_middle = F.softmax(W_middle, dim=1)
        W_last = F.softmax(W_last, dim=1)
        return W_zero, W_first, W_middle, W_last

    def show_weights(self, nb_layer):
        nb_zero_edges = self._nb_zero_edges
        nb_first_edges = self._nb_first_edges
        nb_middle_edges = self._nb_middle_edges
        nb_last_edges = self._nb_last_edges
        return self.normalize_weights(self.alphas_zero_cell[nb_layer * nb_zero_edges: (nb_layer + 1) * nb_zero_edges],
                                      self.alphas_first_cell[
                                      nb_layer * nb_first_edges: (nb_layer + 1) * nb_first_edges],
                                      self.alphas_middle_cell[
                                      nb_layer * nb_middle_edges: (nb_layer + 1) * nb_middle_edges],
                                      self.alphas_last_cell[nb_layer * nb_last_edges: (nb_layer + 1) * nb_last_edges])

    def show_genotype(self, nb_layer):
        outdegree = {}
        gene = []
        nb_zero_nodes = self._nb_zero_nodes
        nb_first_nodes = self._nb_first_nodes
        nb_last_nodes = self._nb_last_nodes
        W_zero, W_first, W_middle, W_last = self.show_weights(nb_layer)

        # amanda@2021/08/30
        # edges in zero cell
        pre_nodes = list(range(0, nb_zero_nodes))
        for n in range(0, nb_zero_nodes):
            k = torch.argmax(W_zero[n]).cpu().item()
            new_node = n + 1
            pre_node = pre_nodes[n]
            gene.append((PRE_OPS[k], new_node, pre_node))
            outdegree[pre_node] = outdegree.get(pre_node, 0) + 1
            pre_nodes[n] = new_node

        # edges in first cell
        start = 0
        for n in range(1, nb_first_nodes + 1):
            end = start + n
            W = W_first[start: end]
            j = sorted(range(n), key=lambda x: -max(W[x][k] for k in range(len(FIRST_OPS))
                                                    if k != FIRST_OPS.index('f_zero')))[0]
            k_best = None
            for k in range(len(FIRST_OPS)):
                if k == FIRST_OPS.index('f_zero'): continue
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            node_id = max(pre_nodes) + n
            pre_node_id = max(pre_nodes) + j
            gene.append((FIRST_OPS[k_best], node_id, pre_node_id))
            outdegree[pre_node_id] = outdegree.get(pre_node_id, 0) + 1
            start = end

        # print(gene)
        # edges in middle cell
        concat_node = []
        middle_nodes = list(range(2, 2 + nb_first_nodes))
        # print(middle_nodes)
        for n in range(0, nb_first_nodes):
            k = torch.argmax(W_middle[n]).cpu().item()
            new_node = max(middle_nodes) + 1
            pre_node = middle_nodes[n]
            gene.append((MIDDLE_OPS[k], new_node, pre_node))
            concat_node.append(new_node)
            outdegree[pre_node] = outdegree.get(pre_node, 0) + 1
            middle_nodes[n] = new_node
            '''if k != MIDDLE_OPS.index('f_identity'):
                new_node = max(middle_nodes) + 1
                pre_node = middle_nodes[n]
                gene.append((MIDDLE_OPS[k], new_node, pre_node))
                outdegree[pre_node] = outdegree.get(pre_node, 0) + 1
                middle_nodes[n] = new_node'''
        # print(pre_node)
        # print(middle_nodes)
        # edges in last cell
        start = 0
        for n in range(nb_last_nodes):
            node_id = n + max(middle_nodes) + 1
            end = start + nb_first_nodes + n
            W = W_last[start: end]
            j = sorted(range(nb_first_nodes + n), key=lambda x: -max(W[x][k] for k in range(len(LAST_OPS))
                                                                     if k != LAST_OPS.index('f_zero')))[0]
            k_best = None
            for k in range(len(LAST_OPS)):
                if k == LAST_OPS.index('f_zero'): continue
                if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
            pre_node_id = middle_nodes[j] if j < nb_first_nodes else j - nb_first_nodes + max(middle_nodes) + 1
            gene.append((LAST_OPS[k_best], node_id, pre_node_id))
            concat_node.append(node_id)
            outdegree[pre_node_id] = outdegree.get(pre_node_id, 0) + 1
            start = end

        # print(gene)
        _genotype = Genotype(alpha_cell=gene,
                             concat_node=concat_node,
                             score_func=None)
        return _genotype

    def show_genotypes(self):
        return [self.show_genotype(i) for i in range(self._layers)]
