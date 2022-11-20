import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl

# configs of three-level search space # 
MIXED_OPS = {
    'pre_mult': lambda args: pre_mult_op(),
    'pre_sub': lambda args: pre_sub_op(),
    'pre_add':lambda args: pre_add_op(),
    'f_zero': lambda args: f_zero_op(),
    'f_identity': lambda args: f_identity_op(),
    'f_dense': lambda args: f_dense_op(args),
    'f_sparse': lambda args: f_sparse_op(args),
    'f_dense_last': lambda args: f_dense_op_last(args),
    'f_sparse_last': lambda args: f_sparse_op_last(args),
    'a_max': lambda args: a_max_op(args),
    'a_mean': lambda args: a_mean_op(args),
    'a_sum': lambda args: a_sum_op(args),
    'a_std': lambda args: a_std_op(args),
}
PRE_OPS = ['pre_mult','pre_sub','pre_add']
FIRST_OPS = ['f_zero', 'f_identity', 'f_dense', 'f_sparse']
#MIDDLE_OPS = ['f_identity', 'a_max', 'a_sum', 'a_mean']
# MIDDLE_OPS = ['a_max']
MIDDLE_OPS = ['a_max', 'a_sum', 'a_mean']
LAST_OPS = ['f_zero', 'f_identity', 'f_dense_last', 'f_sparse_last']
#LAST_OPS = ['f_zero', 'f_identity', 'f_dense', 'f_sparse']

# identify basic h-r feature compute
# fi function

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

def conj(a):
    a[..., 1] = -a[..., 1]
    return a

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

class pre_corr_op(nn.Module):

    def __init__(self):
        super(pre_corr_op, self).__init__()

    def forward(self, g, src_emb, hr):

        return ccorr(src_emb, hr.expand_as(src_emb))

class pre_mult_op(nn.Module):

    def __init__(self):
        super(pre_mult_op, self).__init__()

    def forward(self, g, src_emb, hr):
        return src_emb * hr


class pre_sub_op(nn.Module):

    def __init__(self):
        super(pre_sub_op, self).__init__()

    def forward(self, g, src_emb, hr):
        return src_emb - hr


class pre_add_op(nn.Module):

    def __init__(self):
        super(pre_add_op, self).__init__()

    def forward(self, g, src_emb, hr):
        return src_emb + hr


# global variables & funcations # 

msg = fn.copy_edge('msg_e','m')
EPS = 1e-5
# identity feature filter # 
class f_identity_op(nn.Module):

    def __init__(self):
        super(f_identity_op, self).__init__()

    def forward(self, g, src_emb, src_emb_in):
        return src_emb

# zero feature filter #
class f_zero_op(nn.Module):

    def __init__(self):
        super(f_zero_op, self).__init__()

    def forward(self, g, src_emb, src_emb_in):
        return 0 * src_emb

# max - aggregator #
def reduce_max(nodes):
    accum = torch.max(nodes.mailbox['m'], 1)[0]
    return {'h' : accum}

class a_max_op(nn.Module):

    def __init__(self, args):
        super(a_max_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, block, src_emb, src_emb_in):
        src_emb = F.relu(self.linear(src_emb))
        block.edata['msg_e'] = src_emb
        block.update_all(msg, reduce_max)
        h_node = block.dstdata['h']
        return h_node

# mean - aggregator # 
def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h' : accum}

class a_mean_op(nn.Module):

    def __init__(self, args):
        super(a_mean_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, block, src_emb, src_emb_in):
        src_emb = F.relu(self.linear(src_emb))
        block.edata['msg_e'] = src_emb
        block.update_all(msg, reduce_mean)
        # updating all nodes' embedding according to edge's information in the cell middle
        # without considering dst nodes (...2021/10/04)
        h_node = block.dstdata['h']
        #src_,dst_,_ = block.all_edges(form='all')
        #new_src_emb = h_node[dst_,:]
        #return new_src_emb
        return h_node

# sum - aggregator # 
def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'h': accum}

class a_sum_op(nn.Module):

    def __init__(self, args):
        super(a_sum_op, self).__init__()

    def forward(self, block, src_emb, src_emb_in):
        block.edata['msg_e'] = src_emb
        block.update_all(msg,reduce_sum)
        h_node = block.dstdata['h']
        #src_, dst_, _ = block.all_edges(form='all')
        #new_src_emb = h_node[dst_, :]
        #return new_src_emb
        return h_node


# std - aggregator # 
def reduce_var(h):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var

def reduce_std(nodes):
    h = nodes.mailbox['m']
    return {'h' : torch.sqrt(reduce_var(h) + EPS)}

class a_std_op(nn.Module):

    def __init__(self, args):
        super(a_std_op, self).__init__()

    def forward(self, g, src_emb, src_emb_in):
        g.edata['msg_e'] = src_emb
        g.update_all(msg,reduce_std)
        h_node = g.ndata['h']
        #src_, _, _ = g.all_edges(form='all')
        #new_src_emb = h_node[src_, :]
        #return new_src_emb
        return h_node

# dense - feature filter #
class f_dense_op(nn.Module):
    def __init__(self, args):
        super(f_dense_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim,  self._feature_dim, bias = True)

    def forward(self, g, src_emb, src_emb_in):
        gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(gates)
        return torch.sigmoid(gates) * src_emb

# sparse - feature filter #
class f_sparse_op(nn.Module):
    def __init__(self, args):
        super(f_sparse_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim, self._feature_dim, bias = True)
        self.a = nn.Linear(self._feature_dim, 1, bias = False)

    def forward(self, g, src_emb, src_emb_in):
        gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(gates)
        gates = self.a(gates)
        return torch.sigmoid(gates) * src_emb

# dense - feature filter #
class f_dense_op_last(nn.Module):
    def __init__(self, args):
        super(f_dense_op_last, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(self._feature_dim,  self._feature_dim, bias = True)

    def forward(self, g, src_emb,src_emb_in):
        #gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(src_emb)
        return torch.sigmoid(gates) * src_emb

# sparse - feature filter #
class f_sparse_op_last(nn.Module):
    def __init__(self, args):
        super(f_sparse_op_last, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(self._feature_dim, self._feature_dim, bias = True)
        self.a = nn.Linear(self._feature_dim, 1, bias = False)

    def forward(self, g, src_emb,src_emb_in):
        #gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(src_emb)
        gates = self.a(gates)
        return torch.sigmoid(gates) * src_emb