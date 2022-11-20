import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl

# configs of three-level search space #
MIXED_OPS = {
    'pre_mult': lambda args: pre_mult_op(),
    'pre_sub': lambda args: pre_sub_op(),
    'pre_add': lambda args: pre_add_op(),
    'f_zero': lambda args: f_zero_op(),
    'f_identity': lambda args: f_identity_op(),
    'f_dense': lambda args: f_dense_op(args),
    'f_dense_comp': lambda args: f_dense_op_comp(args),
    'f_comp': lambda args: f_comp_op(args),
    'f_sparse': lambda args: f_sparse_op(args),
    'f_sparse_comp': lambda args: f_sparse_op_comp(args),
    'f_dense_last': lambda args: f_dense_op_last(args),
    'f_sparse_last': lambda args: f_sparse_op_last(args),
    'a_max': lambda args: a_max_op(args),
    'a_mean': lambda args: a_mean_op(args),
    'a_sum': lambda args: a_sum_op(args),
}

MIXED_OPS_sf = {
    'sf_TransE': lambda args: sf_TransE_op(args),
    'sf_DisMult': lambda args: sf_DisMult_op(args),
    'sf_ConvE': lambda args: sf_ConvE_op(args),
}

PRE_OPS = ['pre_mult', 'pre_sub', 'pre_add']
FIRST_OPS = ['f_zero', 'f_identity', 'f_dense_comp', 'f_sparse_comp','f_comp']
MIDDLE_OPS = ['a_max', 'a_sum', 'a_mean']
LAST_OPS = ['f_zero', 'f_identity', 'f_dense_last', 'f_sparse_last']  # for node classification

SF_OPS = ['sf_TransE', 'sf_DisMult']


# SF_OPS = ['sf_TransE']
# SF_OPS = ['sf_DisMult']


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
        # print('hr.shape', hr.shape)
        return src_emb * hr


class pre_sub_op(nn.Module):

    def __init__(self):
        super(pre_sub_op, self).__init__()

    def forward(self, g, src_emb, hr):
        # print('hr.shape', hr.shape)
        return src_emb - hr


class pre_add_op(nn.Module):

    def __init__(self):
        super(pre_add_op, self).__init__()

    def forward(self, g, src_emb, hr):
        # print('hr.shape',hr.shape)
        return src_emb + hr


class sf_TransE_op(nn.Module):

    def __init__(self, args):
        super(sf_TransE_op, self).__init__()
        self.gamma = args.get('gamma', 40)

    def forward(self, all_ent, sub_emb, rel_emb):
        obj_emb = sub_emb + rel_emb
        # x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)
        return score


class sf_DisMult_op(nn.Module):

    def __init__(self, args):
        super(sf_DisMult_op, self).__init__()
        # self.register_parameter('bias', Parameter(torch.zeros(num_en)))

    def forward(self, all_ent, sub_emb, rel_emb):
        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        # self.bias = nn.Parameter(torch.zeros(all_ent.shape[0]))
        # x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class sf_ConvE_op(nn.Module):
    def __init__(self, args):
        """
        :param embed_dim: dimension after second layer
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(sf_ConvE_op, self).__init__()

        self.embed_dim = args.get('embed_dim', 200)
        self.conve_hid_drop, self.feat_drop = args.get('conve_hid_drop', 0.3), args.get('feat_drop',0.3)
        self.num_filt = args.get('num_filt',200)
        self.ker_sz, self.k_w, self.k_h = args.get('ker_sz',7), args.get('k_w',10), args.get('k_h',20)

        #self.conve_hid_drop, self.feat_drop = conve_hid_drop, feat_drop
        #self.num_filt = num_filt
        #self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.feature_drop = torch.nn.Dropout(self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=True)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)  # fully connected projection

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        assert self.embed_dim == self.k_h * self.k_w
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward(self, all_ent, sub_emb, rel_emb):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        stack_input = self.concat(sub_emb, rel_emb)  # [batch_size, 1, 2*k_h, k_w]
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        score = torch.sigmoid(x)
        return score


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


class a_max_op(nn.Module):

    def __init__(self, args):
        super(a_max_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, block, src_emb, src_emb_in):
        src_emb_m = F.relu(self.linear(src_emb[:block.num_edges(), :]))
        block.edata['msg_e'] = src_emb_m
        block.update_all(fn.copy_edge('msg_e', 'm'), fn.max('m', 'h'))
        h_node = block.dstdata['h'] + src_emb[block.num_edges():, :]
        return h_node


class a_mean_op(nn.Module):

    def __init__(self, args):
        super(a_mean_op, self).__init__()
        feature_dim = args.get('feature_dim', 100)
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, block, src_emb, src_emb_in):
        src_emb_m = F.relu(self.linear(src_emb[:block.num_edges(), :]))
        block.edata['msg_e'] = src_emb_m
        block.update_all(fn.copy_edge('msg_e', 'm'), fn.mean('m', 'h'))
        h_node = block.dstdata['h'] + src_emb[block.num_edges():, :]
        return h_node

class a_sum_op(nn.Module):

    def __init__(self, args):
        super(a_sum_op, self).__init__()
        #print(args)
        self.drop_aggr = args.get('drop_aggr', 0.1)
        self.drop_sum = nn.Dropout(self.drop_aggr)

    def forward(self, block, src_emb, src_emb_in):
        block.edata['msg_e'] = src_emb[:block.num_edges(), :]
        block.update_all(fn.copy_edge('msg_e', 'm'), fn.sum('m', 'h'))
        h_node = self.drop_sum(block.dstdata['h']) + src_emb[block.num_edges():, :]
        return h_node

class f_comp_op(nn.Module):
    def __init__(self, args):
        super(f_comp_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W_in = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=False)
        self.W_out = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=False)
        self.W_self = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=False)

    def forward(self, g, src_emb, src_emb_in):
        m_in = torch.cat([src_emb[:g.num_edges() // 2, :], src_emb_in[:g.num_edges() // 2, :]], dim=1)
        m_in = self.W_in(m_in)

        m_out = torch.cat(
            [src_emb[g.num_edges() // 2:g.num_edges(), :], src_emb_in[g.num_edges() // 2:g.num_edges(), :]], dim=1)
        m_out = self.W_out(m_out)

        m_self = torch.cat([src_emb[g.num_edges():, :], src_emb_in[g.num_edges():, :]], dim=1)
        m_self = self.W_self(m_self)
        #print(torch.cat((1 / 3 * m_in, 1 / 3 * m_out), dim=0).shape, g.edata['norm'].unsqueeze(1).shape)
        m_in_out = torch.cat((1 / 3 * m_in, 1 / 3 * m_out), dim=0) * g.edata['norm'].view(-1,1)
        # print(g.num_edges())
        output = torch.cat((m_in_out, m_self), dim=0)
        return output

class f_sparse_op(nn.Module):
    def __init__(self, args):
        super(f_sparse_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)
        self.a = nn.Linear(self._feature_dim, 1, bias=False)

    def forward(self, g, src_emb, src_emb_in):
        gates = torch.cat([src_emb, src_emb_in], dim=1)
        gates = self.W(gates)
        gates = self.a(gates)
        return torch.sigmoid(gates) * src_emb


class f_sparse_op_comp(nn.Module):
    def __init__(self, args):
        super(f_sparse_op_comp, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W_in = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)
        self.a_in = nn.Linear(self._feature_dim, 1, bias=False)

        self.W_out = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)
        self.a_out = nn.Linear(self._feature_dim, 1, bias=False)

        self.W_self = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)
        self.a_self = nn.Linear(self._feature_dim, 1, bias=False)

    def forward(self, g, src_emb, src_emb_in):
        gates_in = torch.cat([src_emb[:g.num_edges() // 2, :], src_emb_in[:g.num_edges() // 2, :]], dim=1)
        gates_in = self.W_in(gates_in)
        gates_in = self.a_in(gates_in)

        gates_out = torch.cat(
            [src_emb[g.num_edges() // 2:g.num_edges(), :], src_emb_in[g.num_edges() // 2:g.num_edges(), :]], dim=1)
        gates_out = self.W_out(gates_out)
        gates_out = self.a_out(gates_out)

        gates_self = torch.cat([src_emb[g.num_edges():, :], src_emb_in[g.num_edges():, :]], dim=1)
        gates_self = self.W_self(gates_self)
        gates_self = self.a_self(gates_self)

        # out_in = torch.sigmoid(gates_in) * src_emb[:g.num_edges() // 2, :] * g.edata['norm'][:g.num_edges() // 2, :]
        # out_o = torch.sigmoid(gates_out) * src_emb[g.num_edges() // 2:g.num_edges(), :] * g.edata['norm'][g.num_edges() // 2:g.num_edges(),:]

        out_in = torch.sigmoid(gates_in) * src_emb[:g.num_edges() // 2, :]
        out_o = torch.sigmoid(gates_out) * src_emb[g.num_edges() // 2:g.num_edges(), :]

        out_self = torch.sigmoid(gates_self) * src_emb[g.num_edges():, :]

        m_in_out = torch.cat((1 / 3 * out_in, 1 / 3 * out_o), dim=0) * g.edata['norm'].view(-1,1)
        m_self = 1 / 3 * out_self
        # print(g.num_edges())
        output = torch.cat((m_in_out, m_self), dim=0)
        return output

class f_dense_op(nn.Module):
    def __init__(self, args):
        super(f_dense_op, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)

    def forward(self, g, src_emb, src_emb_in):
        gates = torch.cat([src_emb, src_emb_in], dim=1)
        gates = self.W(gates)
        return torch.sigmoid(gates) * src_emb

class f_dense_op_comp(nn.Module):
    def __init__(self, args):
        super(f_dense_op_comp, self).__init__()

        self._feature_dim = args.get('feature_dim', 100)
        self.W_in = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)

        self.W_out = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)

        self.W_self = nn.Linear(2 * self._feature_dim, self._feature_dim, bias=True)

    def forward(self, g, src_emb, src_emb_in):
        gates_in = torch.cat([src_emb[:g.num_edges() // 2, :], src_emb_in[:g.num_edges() // 2, :]], dim=1)
        gates_in = self.W_in(gates_in)

        gates_out = torch.cat(
            [src_emb[g.num_edges() // 2:g.num_edges(), :], src_emb_in[g.num_edges() // 2:g.num_edges(), :]], dim=1)
        gates_out = self.W_out(gates_out)

        gates_self = torch.cat([src_emb[g.num_edges():, :], src_emb_in[g.num_edges():, :]], dim=1)
        gates_self = self.W_self(gates_self)

        # out_in = torch.sigmoid(gates_in) * src_emb[:g.num_edges() // 2, :] * g.edata['norm'][:g.num_edges() // 2, :]
        # out_o = torch.sigmoid(gates_out) * src_emb[g.num_edges() // 2:g.num_edges(), :] * g.edata['norm'][g.num_edges() // 2:g.num_edges(),:]

        out_in = torch.sigmoid(gates_in) * src_emb[:g.num_edges() // 2, :]
        out_o = torch.sigmoid(gates_out) * src_emb[g.num_edges() // 2:g.num_edges(), :]

        out_self = torch.sigmoid(gates_self) * src_emb[g.num_edges():, :]

        m_in_out = torch.cat((1 / 3 * out_in, 1 / 3 * out_o), dim=0) * g.edata['norm'].view(-1,1)
        m_self = 1 / 3 * out_self
        # print(g.num_edges())
        output = torch.cat((m_in_out, m_self), dim=0)
        return output
# dense - feature filter for node classification #
class f_dense_op_last(nn.Module):
    def __init__(self, args):
        super(f_dense_op_last, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(self._feature_dim, self._feature_dim, bias=True)

    def forward(self, g, src_emb, src_emb_in):
        # gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(src_emb)
        return torch.sigmoid(gates) * src_emb


# sparse - feature filter for node classification #
class f_sparse_op_last(nn.Module):
    def __init__(self, args):
        super(f_sparse_op_last, self).__init__()
        self._feature_dim = args.get('feature_dim', 100)
        self.W = nn.Linear(self._feature_dim, self._feature_dim, bias=True)
        self.a = nn.Linear(self._feature_dim, 1, bias=False)

    def forward(self, g, src_emb, src_emb_in):
        # gates = torch.cat([src_emb, src_emb_in], dim = 1)
        gates = self.W(src_emb)
        gates = self.a(gates)
        return torch.sigmoid(gates) * src_emb
