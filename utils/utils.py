import os
import shutil

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset


def load_alpha(genotype):
    alpha_cell = torch.Tensor(genotype.alpha_cell)
    alpha_edge = torch.Tensor(genotype.alpha_edge)
    return [alpha_cell, alpha_edge]


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels.

    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(
        scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def gpu_memory(n=0):
    name = torch.cuda.get_device_name(n)
    t = torch.cuda.get_device_properties(n).total_memory
    r = torch.cuda.memory_reserved(n)
    a = torch.cuda.memory_allocated(n)
    f = r - a  # free inside reserved
    res = t, r, a, f
    return [name] + [str(round(x / 1024 / 1024 / 1024, 2)) + ' GB' for x in res]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, fmt=':f'):
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WeightedCE(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(WeightedCE, self).__init__()
        self.n_classes = num_classes
        self.device = device

    def forward(self, pred, label):
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[torch.nonzero(label_count, as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # weighted cross-entropy for unbalanced classes
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)
        return loss


def load_batch(args, device):
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    # Load from hetero-graph
    hg = dataset[0]
    hg = hg.to(device)
    # print(hg.number_of_nodes())
    # print(hg.number_of_edges())

    '''
    # Obtain the basic information of  a graph, containing: 
    1. the number of relations；
    2. the number of classes for prediction;
    3. train_idx, test_idx, and val_idx(1/5 train_idx)；
    4. labels of all nodes, if not in predict_category, label as '-1'
    '''
    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')
    # split dataset into train, validate, test
    if args.valid:
        # search architecture
        val_idx = train_idx[:len(train_idx) // 2]
        train_idx = train_idx[len(train_idx) // 2:]
    else:
        # train from scratch, no split
        val_idx = train_idx

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    '''
    Converting the loaded hetero graph into homogeneous one, where each node's and edge's id is redefined,with:
    1. node_ids
    2. calculated edge_norm
    3. current edge_type;
    '''
    g = dgl.to_homogeneous(hg)

    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.ndata['type_id'] = g.ndata[dgl.NID]

    """make a triple_g for indexing original src-dst nodes and edges"""
    src_g, dst_g, eid_g = g.edges(form='all')
    triple_g = torch.stack((eid_g, src_g, dst_g), dim=1)

    # find out the target node ids
    node_ids = torch.arange(g.number_of_nodes(), device=device)
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    # 237 nodes, target_idx为目标节点在同构图g上的id

    """
    This is a graph with multiple node types, so we want a way to map
    # our target node from their global node numberings, back to their
    # numberings within their type. This is used when taking the nodes in a
    # mini-batch, and looking up their type-specific labels
    # inv_target 映射回异构图hg的类内id
    """
    inv_target = torch.empty(node_ids.shape, dtype=node_ids.dtype)
    inv_target[target_idx] = torch.arange(0, target_idx.shape[0], dtype=inv_target.dtype)

    return triple_g, g, train_idx, target_idx, val_idx, test_idx, labels, num_rels, num_classes, inv_target


def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    print({'day': t, 'hour': h, 'minute': m, 'second': int(s)})
    return f"{int(t)}:{int(h)}:{int(m)}"


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D

    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.

    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
