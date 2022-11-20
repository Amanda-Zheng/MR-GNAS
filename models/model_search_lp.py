import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from configs.genotypes import Genotype
from models.operations_lp import PRE_OPS, FIRST_OPS, MIDDLE_OPS, LAST_OPS, SF_OPS
from models.cell_lp import Cell, Cell_SF
from functools import partial
import dgl.function as fn

from utils.gpu_memory_log import gpu_memory_log
from utils.utils import count_parameters_in_MB


class Network(nn.Module):
    def __init__(self, device, number_of_nodes, num_rels, layers, zero_nodes, first_nodes, last_nodes, feature_dim,
                 init_fea_dim,
                 num_base_r, gamma, dropout_cell, drop_aggr):
        super(Network, self).__init__()
        self._device = device
        self._layers = layers
        self._num_ent = number_of_nodes
        self._num_rel = num_rels * 2 + 1
        self._feature_dim = feature_dim
        self.num_base_r = num_base_r
        self.init_fea_dim = init_fea_dim
        self._nb_zero_nodes = zero_nodes
        self._nb_first_nodes = first_nodes
        self._nb_last_nodes = last_nodes
        self._nb_zero_edges = self._nb_zero_nodes

        # final nodes and edges are always equal to 1 for score function
        self._nb_final_nodes = 1
        self._nb_final_edges = self._nb_final_nodes
        #
        self._nb_first_edges = sum(self._nb_zero_nodes + i for i in range(self._nb_first_nodes))
        self._nb_middle_edges = self._nb_first_nodes
        self._nb_last_edges = sum(self._nb_first_nodes + i for i in range(self._nb_last_nodes))

        self.embedding_h = nn.Embedding(self._num_ent, self.init_fea_dim)  # node feat is an integer
        self.embedding_e = nn.Embedding(self.num_base_r, self._feature_dim)

        self.linear_e = nn.Linear(self.init_fea_dim, self._feature_dim)

        '''
        # here is for the v1 version 
        {
        self.linear_e_init = nn.Linear(self.init_fea_dim, self._feature_dim)
        self.linear_r_init = nn.Linear(self.init_fea_dim, self._feature_dim)
        self.linear_r_final = nn.Linear(self._feature_dim, self._feature_dim)
        }
        '''
        # self.conv_r  = torch.nn.Conv2d(self.num_base_r, self._num_rel, (1,1), stride=1)
        self.idx_ent = torch.arange(self._num_ent, device=self._device)
        self.idx_rel = torch.arange(self.num_base_r, device=self._device)
        # print("param size = %fMB", count_parameters_in_MB(self)

        self.rel_wt = self.get_param([self._num_rel, self.num_base_r])

        self.w_rel = self.get_param(
            [self._feature_dim, self._feature_dim])

        self._drop_aggr = drop_aggr

        self.cells = nn.ModuleList(
            [Cell(self._nb_zero_nodes, self._nb_first_nodes, self._nb_last_nodes, self._feature_dim, self._drop_aggr)
             for i in range(self._layers)])
        self._initialize_alphas()

        self.gamma = gamma
        self.score_func = Cell_SF(self.gamma)

        self.batchnorm_h = nn.BatchNorm1d(self._feature_dim)
        self.activate = nn.ReLU()
        self._dropout = dropout_cell

    '''
    def new(self):
        model_new = Network(self._device, self._num_ent, self._layers, self._nb_zero_nodes,
                            self._nb_first_nodes, self._feature_dim).to(self._device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    '''

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
        nb_final_edges = self._nb_final_edges
        nb_zero_ops = len(PRE_OPS)
        nb_first_ops = len(FIRST_OPS)
        nb_middle_ops = len(MIDDLE_OPS)
        nb_last_ops = len(LAST_OPS)
        nb_final_ops = len(SF_OPS)

        self.alphas_zero_cell = Variable(1e-3 * torch.randn(nb_zero_edges * nb_layers, nb_zero_ops).to(self._device),
                                         requires_grad=True)
        self.alphas_first_cell = Variable(1e-3 * torch.randn(nb_first_edges * nb_layers, nb_first_ops).to(self._device),
                                          requires_grad=True)
        self.alphas_middle_cell = Variable(
            1e-3 * torch.randn(nb_middle_edges * nb_layers, nb_middle_ops).to(self._device), requires_grad=True)
        self.alphas_last_cell = Variable(1e-3 * torch.randn(nb_last_edges * nb_layers, nb_last_ops).to(self._device),
                                         requires_grad=True)
        self.alphas_final_cell = Variable(1e-3 * torch.randn(nb_final_edges, nb_final_ops).to(self._device),
                                          requires_grad=True)

        self._arch_parameters = [
            self.alphas_zero_cell,
            self.alphas_first_cell,
            self.alphas_middle_cell,
            self.alphas_last_cell,
            self.alphas_final_cell
        ]

    def _forward_lp(self, g_train, node_id, src_in, edge_type):

        all_ent_emb = self.linear_e(self.embedding_h(self.idx_ent))
        rel_embed = torch.mm(self.rel_wt, self.embedding_e(self.idx_rel))
        src_in_final = torch.cat((src_in, g_train.nodes()), dim=0)
        edge_self = (torch.ones(g_train.nodes().shape) * (self._num_rel - 1)).long().to(self._device)

        src_id_final = node_id[src_in_final].squeeze()
        edge_type_final = torch.cat((edge_type, edge_self), dim=0)

        for i, cell in enumerate(self.cells):
            W_zero, W_first, W_middle, W_last = self.show_weights(i)
            if i == 0:
                ent_emb_in = all_ent_emb[src_id_final]
                ent_emb = cell(g_train, ent_emb_in, rel_embed[edge_type_final], W_zero, W_first, W_middle, W_last)
                ent_emb = self.batchnorm_h(ent_emb)
                if len(self.cells) == 1:
                    ent_emb = self.activate(ent_emb)
                ent_emb = F.dropout(ent_emb, self._dropout, training=self.training)
                rel_embed = torch.matmul(rel_embed, self.w_rel)

            else:
                ent_emb_in = torch.cat((ent_emb[src_in], ent_emb),dim=0)
                ent_emb = cell(g_train, ent_emb_in, rel_embed[edge_type_final], W_zero, W_first, W_middle, W_last)
                ent_emb = self.batchnorm_h(ent_emb)
                ent_emb = self.activate(ent_emb)
                ent_emb = F.dropout(ent_emb, self._dropout, training=self.training)
                rel_embed = torch.matmul(rel_embed, self.w_rel)

        # W_final = F.softmax(self.alphas_final_cell[:self._nb_final_edges], dim=1)
        # score_f = self.score_func(all_ent_emb, all_ent_emb[subj], rel_embed[rel], W_final)

        return ent_emb, rel_embed

    def forward(self, g_train, node_id, src_in, edge_type):
        batch_ent_emb, batch_rel_embed = self._forward_lp(g_train, node_id, src_in, edge_type)
        return batch_ent_emb, batch_rel_embed

    def calc_score(self, ent_embedding, rel_embedding, triplets):
        # DistMult
        # print('embedding.shape', embedding.shape)
        s = ent_embedding[triplets[:, 0]]
        r = rel_embedding[triplets[:, 1]]
        o = ent_embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g_train, ent_embed, rel_embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(ent_embed, rel_embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        # reg_loss = self.regularization_loss(embed)
        # return predict_loss + self.reg_param * reg_loss
        return predict_loss

    def _loss(self, g_train, node_id, src_in, edge_type, triplets, labels):
        batch_ent_emb, batch_rel_emb = self.forward(g_train, node_id, src_in, edge_type)
        score = self.calc_score(batch_ent_emb, batch_rel_emb, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss

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
        nb_final_nodes = self._nb_final_nodes
        W_zero, W_first, W_middle, W_last = self.show_weights(nb_layer)
        # W_final = F.softmax(self.alphas_final_cell[:self._nb_final_edges], dim=1)

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
            # if k != MIDDLE_OPS.index('f_identity'):
            new_node = max(middle_nodes) + 1
            pre_node = middle_nodes[n]
            gene.append((MIDDLE_OPS[k], new_node, pre_node))
            concat_node.append(new_node)
            outdegree[pre_node] = outdegree.get(pre_node, 0) + 1
            middle_nodes[n] = new_node
            # need to make sure ops!= identity
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

        # edges in final cell
        score_func = None
        '''if nb_layer == self._layers - 1:
            for n in range(0, nb_final_nodes):
                k = torch.argmax(W_final[n]).cpu().item()
                new_node = n + max(concat_node) + 1
                pre_node = concat_node[-1]
                score_func = SF_OPS[k]
                # gene.append((SF_OPS[k], new_node, pre_node))
                # concat_node.append(new_node)
                # outdegree[pre_node] = outdegree.get(pre_node, 0) + 1'''

        _genotype = Genotype(alpha_cell=gene,
                             concat_node=concat_node,
                             score_func=score_func)
        return _genotype

    def show_genotypes(self):
        return [self.show_genotype(i) for i in range(self._layers)]
