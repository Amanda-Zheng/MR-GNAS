import time
import numpy as np
import random
import argparse
import datetime
import logging
import sys
from tensorboardX import SummaryWriter

sys.path.append('..')
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
import dgl
import torch.nn.functional as F
from dataloader import get_dataset
from dgl.contrib.data import load_data
from models.model_search_lp import Network
from models.architect_lp import Architect

from utils.process_data import process
from utils.data_set import TestDataset, TrainDataset
from utils.utils import *
import utils.utils_rgcn as utils_rgcn


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    # g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    return g.edata['norm']

def load_kg(args,device):
    if args.dataset == 'FB15k-237':
        dataset =dgl.data.FB15kDataset()
        graph = dataset[0]
        train_mask = graph.edata['train_mask']
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        src, dst = graph.edges(train_idx)
        rel = graph.edata['etype'][train_idx]


def load_kg_pos_neg(args, device):
    # load graph data
    if args.dataset == 'FB15k-237':
        data = load_data(args.dataset)
        num_nodes, train_data, valid_data, test_data, num_rels = data.num_nodes, data.train, data.valid, data.test, data.num_rels
    elif args.dataset == 'wn18rr':
        data = get_dataset('data', 'wn18rr', 'built_in', None)
        num_nodes, train_data, valid_data, test_data, num_rels = data.n_entities, np.array(
            data.train).transpose(), np.array(data.valid).transpose(), np.array(data.test).transpose(), data.n_relations
    else:
        print('Unknown Dataset, please check.')

    # valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_src_in, test_rel, test_norm = utils_rgcn.build_test_graph(
        num_nodes, num_rels, train_data)
    # val_graph, val_src_in, val_rel, val_norm = utils_rgcn.build_test_graph(
    #    num_nodes, num_rels, valid_data)
    # val_graph has 14541 nodes, and 35070 edges，so the sample edges can not be over 3w
    # wn18, val_graph has 40943nodes, 10000 edges, so val do not need to sample

    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
    test_graph.edata['norm'] = test_norm

    # val_norm = node_norm_to_edge_norm(val_graph, torch.from_numpy(val_norm).view(-1, 1))
    # val_graph.edata['norm'] = val_norm
    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils_rgcn.get_adj_and_degrees(num_nodes, train_data)
    adj_list_val, degrees_val = utils_rgcn.get_adj_and_degrees(num_nodes, valid_data)
    #print(adj_list)
    return train_data, num_rels, num_nodes, adj_list, degrees, adj_list_val, degrees_val, valid_data, \
           test_data, test_graph, test_src_in, test_rel, test_node_id


def start_batch(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    max_patience = args.max_patience
    patience = 0
    warm_epochs = args.warm_epochs
    cudnn.benchmark = True
    cudnn.enabled = True
    device = torch.device(args.device)
    start_time = time.time()
    logging.info('RUNDIR: {}'.format(log_dir))
    logging.info("args = %s", args)
    train_data, num_rels, num_ent, adj_list, degrees, \
    adj_list_val, degrees_val, valid_data, \
    test_data, test_graph, test_src_in, test_rel, test_node_id = load_kg_pos_neg(args, device)

    # Model definition
    model = Network(device, num_ent, num_rels, args.layers, args.zero_nodes, args.first_nodes,
                    args.last_nodes, args.feature_dim, args.init_fea_dim, args.num_base_r, args.gamma, args.dropout_cell, args.drop_aggr)

    model = model.to(device)
    # print(model)
    model.apply(weights_init)
    best_geno = str(model.show_genotypes())

    # Calculating model's parameters
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    # Define optimizer for model updating & lr deccay for better learning

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=float(args.epochs // args.lr_var), gamma=0.1)

    # Define searched architecture
    architect = Architect(device, model, args)

    for epoch in range(args.epochs):

        if epoch % args.save_freq == 0:
            logging.info(model.show_genotypes())
            for i in range(args.layers):
                logging.info('layer = %d', i)
                genotype = model.show_genotype(i)
                logging.info('genotype = %s', genotype)
        # training
        print('begin to train...')

        best_mrr = 0
        best_epoch_mrr = 0

        train_loss, archi_loss = train(train_data, num_rels, adj_list, degrees, valid_data, adj_list_val, degrees_val,
                                       model, architect, optimizer, scheduler, epoch, warm_epochs, device)

        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Arch/loss', archi_loss, epoch)

        # Do not testing when searching

        # @ Here are for testing

        '''

        mrr, hit_k = infer_graph(test_graph, test_src_in, test_rel, test_node_id, model, train_data, valid_data, test_data,device)
        # logging.info('test_head_results', test_head_results)
        # writer.add_scalar('Test/loss', test_loss, epoch)
        # writer.add_scalar('Test/mr', test_results['mr'], epoch)
        writer.add_scalar('Test/mrr', mrr, epoch)
        writer.add_scalar('Test/hits_1', hit_k[0], epoch)
        writer.add_scalar('Test/hits_3', hit_k[1], epoch)
        writer.add_scalar('Test/hits_10', hit_k[2], epoch)

        if best_mrr < mrr:
            best_mrr = mrr
            best_epoch_mrr = epoch

        if epoch % 10 == 0:
            logging.info("Best_epoch_mrr {:04d} mrr {:.3f}".format(best_epoch_mrr, best_mrr))
        '''

        if epoch >= warm_epochs:
            # check whether the genotype has changed
            genotypes = str(model.show_genotypes())
            if best_geno == genotypes:
                patience += 1
            else:
                patience = 0
                best_geno = genotypes

            logging.info('Current patience :{}'.format(patience))
            if patience >= max_patience:
                logging.info('Reach the max patience! \n best genotype {}'.format(best_geno))
                break
    logging.info('Duration is {} (day:h:min)'.format(calc_time(time.time() - start_time)))
    logging.info('All done! \n best genotype {}'.format(best_geno))
    logging.info('RUNDIR: {}'.format(log_dir))


def train(train_data, num_rels, adj_list, degrees, valid_data, adj_list_val, degrees_val, model,
          architect, optimizer, scheduler, epoch, warm_epochs, device):
    lr = scheduler.get_last_lr()[-1]
    logging.info('[LR]\t%f', lr)
    model.train()

    g, node_id, src_in, edge_type, node_norm, data, labels = \
        utils_rgcn.generate_sampled_graph_and_labels(train_data, args.graph_batch_size, args.graph_split_size,
                                                     num_rels, adj_list, degrees, args.negative_sample,
                                                     args.edge_sampler)

    '''
    if args.dataset == 'FB15k-237':
        val_graph_bs = args.graph_batch_size // 2
    else:
        val_graph_bs = args.graph_batch_size // 6
        '''

    g_val, node_id_val, src_in_val, edge_type_val, node_norm_val, data_val, labels_val = \
        utils_rgcn.generate_sampled_graph_and_labels(valid_data, args.graph_batch_size_val, args.graph_split_size,
                                                     num_rels, adj_list_val, degrees_val, args.negative_sample,
                                                     args.edge_sampler)
    logging.info("Done edge sampling")

    # set node/edge feature for train & validation
    node_id = torch.from_numpy(node_id).view(-1, 1).long().to(device)
    src_in = torch.from_numpy(src_in).to(device)
    edge_type = torch.from_numpy(edge_type).to(device)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1)).to(device)
    data, labels = torch.from_numpy(data).to(device), torch.from_numpy(labels).to(device)
    g_train = g.to(device)
    g_train.edata['norm'] = edge_norm

    node_id_val = torch.from_numpy(node_id_val).view(-1, 1).long().to(device)
    src_in_val = torch.from_numpy(src_in_val).to(device)
    edge_type_val = torch.from_numpy(edge_type_val).to(device)
    edge_norm_val = node_norm_to_edge_norm(g_val, torch.from_numpy(node_norm_val).view(-1, 1)).to(device)
    data_val, labels_val = torch.from_numpy(data_val).to(device), torch.from_numpy(labels_val).to(device)
    g_val = g_val.to(device)
    g_val.edata['norm'] = edge_norm_val

    t0 = time.time()

    if epoch >= warm_epochs:
        architect.step(g_train, node_id, src_in, edge_type, data, labels,
                       g_val, node_id_val, src_in_val, edge_type_val, data_val, labels_val,
                       optimizer, lr, unrolled=args.unrolled)

    ent_embed, rel_embed = model(g_train, node_id, src_in, edge_type)

    # if epoch == 0:
    #    logging.info('GPU {} memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

    loss = model.get_loss(g_train, ent_embed, rel_embed, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
    optimizer.step()
    scheduler.step()
    t2 = time.time()

    forward_time = t1 - t0
    backward_time = t2 - t1
    logging.info("Epoch {:04d} | Loss {:.4f} | Arch_Loss {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
                 format(epoch, loss.item(), architect.loss.item(), forward_time, backward_time))

    optimizer.zero_grad()

    return loss.item(), architect.loss.item()


def infer_graph(test_graph, test_src_in, test_rel, test_node_id, model, train_data, valid_data, test_data, device):
    # validation
    if True:
        # perform validation on CPU because full graph is too large
        # if args.device != 'cpu':
        #    model.cpu()
        model.eval()
        logging.info("start eval")
        # test_node_id = torch.from_numpy(test_node_id).view(-1, 1).long().to(device)
        test_src_in = torch.from_numpy(test_src_in).to(device)
        test_rel = test_rel.to(device)
        test_graph = test_graph.to(device)
        ent_embed, rel_embed = model(test_graph, test_node_id, test_src_in, test_rel)
        mrr, hit_k = utils_rgcn.calc_mrr(ent_embed, rel_embed, torch.LongTensor(train_data),
                                         torch.LongTensor(valid_data), test_data, hits=[1, 3, 10],
                                         eval_bz=args.eval_batch_size,
                                         eval_p=args.eval_protocol)

        # if args.device != 'cpu':
        #    model.gpu()
        return mrr, hit_k


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MG-NAS-LP")
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='Dataset to use, default: FB15k-237，wn18rr')
    parser.add_argument('--layers', type=int, default=2, help='total number of layers, 2')
    parser.add_argument('--init_fea_dim', type=int, default=100, help='number of initial embedding dim of features')
    parser.add_argument('--feature_dim', type=int, default=200, help='number of features')
    parser.add_argument('--num_base_r', type=int, default=475, help='relation-basis composition {475,37}')
    parser.add_argument('--zero_nodes', type=int, default=1, help='nodes in zero cell with edges information')
    parser.add_argument('--first_nodes', type=int, default=2, help='nodes in first cell')
    parser.add_argument('--last_nodes', type=int, default=2, help='nodes in last cell, 2')
    parser.add_argument('--epochs', type=int, default=8000, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-5, help='weight decay for arch encoding')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-5, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=0e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--dropout_cell', type=float, default=0.3, help='dropout out of cell for training')
    parser.add_argument('--drop_aggr', type=float, default=0.1, help='dropout out in aggregation func')

    parser.add_argument('--gamma', type=float, default=40, help='margin for transE score func')
    parser.add_argument('--lr_var', type=int, default=1, help='changing lr decay rate by epoch percentatge')

    parser.add_argument('--save_freq', type=int, default=5)
    # parser.add_argument('--save_result', type=str, default="./save/Current_search.txt")
    # parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.0, help='Label Smoothing')
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader.")
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cpu')
    parser.add_argument('--max_patience', type=int, default=500, help='Max patience for early stopping')
    parser.add_argument('--warm_epochs', type=int, default=10, help='Warm up epochs to freeze architecture params')
    # parser.add_argument('--valid', type=bool, default=True, help='whether split train/valid/test')

    parser.add_argument("--graph_batch_size", type=int, default=300,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph_batch_size_val", type=int, default=100,
                        help="number of edges to sample in each iteration for validation {fb25k237:10000; wn18:80000")
    parser.add_argument("--graph_split_size", type=float, default=0.5, help="portion of edges used as positive sample")
    parser.add_argument("--negative_sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--edge_sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--grad_norm", type=float, default=5.0, help="norm to clip gradient to")

    parser.add_argument("--eval_batch_size", type=int, default=1000, help="batch size when evaluating")
    parser.add_argument("--eval_protocol", type=str, default="filtered",
                        help="type of evaluation protocol: 'raw' or 'filtered' mrr")

    # ConvE specific hyperparameters
    '''
    parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.2, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--input_drop', dest='input_drop', default=0.2, type=float, help='ConvE: Stacked Input Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')'''

    args = parser.parse_args()

    log_dir = './' + args.save + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    writer = SummaryWriter(log_dir + '/tbx_log')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'search.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start_batch(args)
