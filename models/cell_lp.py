import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.autograd import Variable
from configs.genotypes import Genotype
from models.operations_lp import PRE_OPS, FIRST_OPS, MIDDLE_OPS, LAST_OPS, MIXED_OPS, MIXED_OPS_sf, SF_OPS
from utils.gpu_memory_log import gpu_memory_log


class MixedOp(nn.Module):
    def __init__(self, feature_dim, drop_aggr, operations):
        super(MixedOp, self).__init__()
        self._feature_dim = feature_dim
        self._operations = operations
        self._drop_aggr = drop_aggr

        self._args = {'feature_dim': self._feature_dim, 'drop_aggr': self._drop_aggr}
        self._ops = nn.ModuleList([nn.ModuleList([MIXED_OPS[op_name](self._args),
                                                  nn.BatchNorm1d(self._feature_dim),
                                                  nn.ReLU()]) for op_name in self._operations])
        #print(self._ops)

    def forward(self, weights, g, h, h_in):
        output = sum(w * self.op_forward(op, g, h, h_in) for w, op in zip(weights, self._ops))
        return output

    def op_forward(self, op, g, h, h_in):
        nh = op[0](g, h, h_in)
        for i in range(1, len(op)):
            nh = op[i](nh.float())
        return nh


class MixedOp_SF(nn.Module):
    def __init__(self, gamma, operations):
        super(MixedOp_SF, self).__init__()
        self._operations = operations
        self.gamma = gamma
        self._args = {'gamma': self.gamma}
        self._ops = nn.ModuleList([nn.ModuleList([MIXED_OPS_sf[op_name](self._args)]) for op_name in self._operations])

    def forward(self, weights, g, h, h_in):
        output = sum(w * self.op_forward(op, g, h, h_in) for w, op in zip(weights, self._ops))
        return output

    def op_forward(self, op, g, h, h_in):
        nh = op[0](g, h, h_in)
        return nh


class Cell_Zero(nn.Module):

    def __init__(self, nodes, feature_dim, drop_aggr):
        super(Cell_Zero, self).__init__()
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        mixed_op = MixedOp(feature_dim, drop_aggr, operations=PRE_OPS)
        self._ops.append(mixed_op)

    def forward(self, g, h, hr, weights):
        '''
        states  len : nb_nodes
        weights len : nb_nodes * nb_operations
        '''
        output = self._ops[0](weights[0], g, h, hr)
        return output


class Cell_Final(nn.Module):

    def __init__(self, gamma):
        super(Cell_Final, self).__init__()
        # self.gamma = gamma
        self._ops = nn.ModuleList()
        mixed_op = MixedOp_SF(gamma, operations=SF_OPS)
        self._ops.append(mixed_op)

    def forward(self, all_ent, sub_emb, rel_emb, weights):
        '''
        states  len : nb_nodes
        weights len : nb_nodes * nb_operations
        '''
        output = self._ops[0](weights[0], all_ent, sub_emb, rel_emb)
        return output


class Cell_First(nn.Module):

    def __init__(self, nodes, feature_dim, drop_aggr):
        super(Cell_First, self).__init__()
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            for j in range(i + 1):  # 1 + previous nodes sum
                mixed_op = MixedOp(feature_dim, drop_aggr, operations=FIRST_OPS)
                self._ops.append(mixed_op)

    def forward(self, g, states, h_in, weights):
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](weights[offset + j], g, h, h_in) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states[1:]


class Cell_Middle(nn.Module):

    def __init__(self, nodes, feature_dim, drop_aggr):
        super(Cell_Middle, self).__init__()
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            self._ops.append(MixedOp(feature_dim, drop_aggr, operations=MIDDLE_OPS))

    def forward(self, g, states, h_in, weights):
        '''
        states  len : nb_nodes
        weights len : nb_nodes * nb_operations
        '''
        output = [self._ops[i](weights[i], g, states[i], h_in) for i in range(self._nodes)]
        return output


class Cell_Last(nn.Module):

    def __init__(self, in_nodes, nodes, feature_dim, drop_aggr):
        super(Cell_Last, self).__init__()
        self._in_nodes = in_nodes
        self._nodes = nodes
        self._feature_dim = feature_dim
        self._ops = nn.ModuleList()
        for i in range(nodes):
            for j in range(i + in_nodes):
                mixed_op = MixedOp(feature_dim, drop_aggr, operations=LAST_OPS)
                self._ops.append(mixed_op)

    def forward(self, g, states, h_in, weights):
        '''
        states len : nb_in_nodes
        '''
        offset = 0
        for i in range(self._nodes):
            s = sum(self._ops[offset + j](weights[offset + j], g, h, h_in) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return states


class Cell(nn.Module):

    def __init__(self, nb_zero_nodes, nb_first_nodes, nb_last_nodes, feature_dim, dropout_aggr):
        super(Cell, self).__init__()
        self._nb_zero_nodes = nb_zero_nodes
        self._nb_first_nodes = nb_first_nodes
        self._nb_last_nodes = nb_last_nodes
        self._feature_dim = feature_dim
        # self._drop_aggr = dropout_aggr
        self.cell_zero = Cell_Zero(nb_zero_nodes, feature_dim, dropout_aggr)
        self.cell_first = Cell_First(nb_first_nodes, feature_dim, dropout_aggr)
        self.cell_middle = Cell_Middle(nb_first_nodes, feature_dim, dropout_aggr)
        self.cell_last = Cell_Last(nb_first_nodes, nb_last_nodes, feature_dim, dropout_aggr)
        # self.concat_weights = nn.Linear((nb_zero_nodes + nb_first_nodes + nb_last_nodes) * feature_dim, feature_dim)
        self.concat_weights = nn.Linear((nb_first_nodes + nb_last_nodes) * feature_dim, feature_dim)
        #self.batchnorm_h = nn.BatchNorm1d(feature_dim)
        #self.activate = nn.ReLU()

    def forward(self, g, src_emb, hr, weights_zero, weights_first, weights_middle, weights_last):
        h_in = self.cell_zero(g, src_emb, hr, weights_zero)

        states = [h_in]

        states = self.cell_first(g, states, h_in, weights_first)

        states = self.cell_middle(g, states, h_in, weights_middle)

        states = self.cell_last(g, states, h_in, weights_last)

        new_states = states

        h = self.concat_weights(torch.cat(new_states, dim=1))

        return h


class Cell_SF(nn.Module):

    def __init__(self, gamma):
        super(Cell_SF, self).__init__()
        # self._nb_SF_nodes = nb_SF_nodes
        self.cell_score = Cell_Final(gamma)

    def forward(self, all_ent_emb, sub_emb, rel_emb, weights_sf):
        score = self.cell_score(all_ent_emb, sub_emb, rel_emb, weights_sf)
        return score
