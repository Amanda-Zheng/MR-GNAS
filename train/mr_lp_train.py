# This is for link prediction for datasets FB15K237, WN18RR.
# author: Amanda
# date: 2021/11
import datetime
import sys
import os
import dgl
import time
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

sys.path.append('..')

from utils.utils import *
from tqdm import tqdm
from models.model_lp import Network
from configs.genotypes import *
from configs import genotypes
import torch.backends.cudnn as cudnn
from dgl.contrib.data import load_data
from utils.process_data import process
from utils.data_set import TestDataset, TrainDataset
from utils.utils import *
from torch.utils.data import DataLoader
from dataloader import get_dataset
from utils.gpu_memory_log import gpu_memory_log


def load_kg(args, device):
    if args.dataset == 'FB15k-237':
        data = load_data(args.dataset)
        num_ent, train_data, valid_data, test_data, num_rels = data.num_nodes, data.train, data.valid, data.test, data.num_rels
    elif args.dataset == 'wn18rr':
        data = get_dataset('data', 'wn18rr', 'built_in', None)
        num_ent, train_data, valid_data, test_data, num_rels = data.n_entities, np.array(
            data.train).transpose(), np.array(data.valid).transpose(), np.array(data.test).transpose(), data.n_relations
    else:
        print('Unknown Dataset, please check.')
    triplets = process({'train': train_data, 'valid': valid_data, 'test': test_data}, num_rels)
    g_train = build_graph(num_ent, train_data, num_rels).to(device)

    train_loader = DataLoader(TrainDataset(triplets['train'], num_ent, args),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_tail_loader = DataLoader(TestDataset(triplets['train_tail'], num_ent, args),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_head_loader = DataLoader(TestDataset(triplets['train_head'], num_ent, args),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    valid_loader = DataLoader(TrainDataset(triplets['valid_head'] + triplets['valid_tail'], num_ent, args),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_head_loader = DataLoader(TestDataset(triplets['valid_head'], num_ent, args), batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    valid_tail_loader = DataLoader(TestDataset(triplets['valid_tail'], num_ent, args), batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    test_head_loader = DataLoader(TestDataset(triplets['test_head'], num_ent, args),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_tail_loader = DataLoader(TestDataset(triplets['test_tail'], num_ent, args),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return g_train, triplets, num_ent, num_rels, \
           train_loader, train_tail_loader, train_head_loader, \
           valid_loader, valid_head_loader, valid_tail_loader, \
           test_head_loader, test_tail_loader


# g build on train data;
def build_graph(num_ent, data, num_rels):
    g = dgl.DGLGraph()
    g.add_nodes(num_ent)
    g.add_edges(data[:, 0], data[:, 2])
    g.add_edges(data[:, 2], data[:, 0])
    in_deg = g.in_degrees(range(g.number_of_nodes())).cpu().float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    g.ndata['n_norm'] = torch.tensor(norm)
    g.apply_edges(lambda edges: {'norm': edges.dst['n_norm'] * edges.src['n_norm']})
    edge_type = torch.tensor(np.concatenate([data[:, 1], data[:, 1] + num_rels]))
    g.edata['e_type'] = edge_type
    return g


def start_batch(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    device = torch.device(args.device)
    start_time = time.time()
    logging.info('RUNDIR: {}'.format(log_dir))
    logging.info("args = %s", args)

    g_train, triplets, num_ent, num_rels, \
    train_loader, train_tail_loader, train_head_loader,\
    valid_loader, valid_head_loader, valid_tail_loader, \
    test_head_loader, test_tail_loader = load_kg(args, device)

    if len(args.genotype) > 0:
        genotype = eval(args.genotype)
        logging.info('=> loading genotype: {}'.format(genotype))
    else:
        logging.info("Unknown genotype.")
        exit()

    criterion = torch.nn.BCELoss()
    # criterion = WeightedCE(num_classes, device)
    criterion = criterion.to(device)

    # Model definition
    model = Network(device, genotype, num_ent, num_rels, args.feature_dim, args.init_fea_dim, args.num_base_r,
                    criterion, args.dropout_cell, args)

    model = model.to(device)

    if len(args.checkpoint) != 0:
        # checkpoint = torch.load(args.checkpoint,map_location ='cpu')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("loading checkpoint from epoch: {}".format(checkpoint['epoch']))
    else:
        model.apply(weights_init)

    logging.info("param size = %fMB", count_parameters_in_MB(model))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
        #                                                       patience=5, verbose=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=float(args.epochs), gamma=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
                                                           eta_min=args.learning_rate_min)

    best_mrr = 0
    best_hits_1 = 0
    best_hits_10 = 0
    best_epoch_mrr = 0
    best_epoch_hits = 0
    for epoch in range(args.epochs):
        logging.info('[EPOCH]\t%d', epoch)

        #if args.optimizer == 'SGD':
        lr = scheduler.get_last_lr()[-1]
        logging.info('[LR]={:.06f}'.format(lr))

        train_loss = train(train_loader, g_train, model, optimizer, scheduler, epoch, criterion, args.save_model_freq, device)


        writer.add_scalar('Train/loss', train_loss, epoch)

        val_results, val_loss = infer(model, g_train, valid_tail_loader, valid_head_loader, epoch, 'val', device)

        # validation
        writer.add_scalar('Valid/loss', val_loss, epoch)
        writer.add_scalar('Valid/mr', val_results['mr'], epoch)
        writer.add_scalar('Valid/mrr', val_results['mrr'], epoch)
        writer.add_scalar('Valid/hits_1', val_results[f'hits@{1}'], epoch)
        writer.add_scalar('Valid/hits_3', val_results[f'hits@{3}'], epoch)
        writer.add_scalar('Valid/hits_10', val_results[f'hits@{10}'], epoch)

        # testing
        test_results, test_loss = infer(model, g_train, test_tail_loader, test_head_loader, epoch, 'test', device)
        writer.add_scalar('Test/loss', test_loss, epoch)
        writer.add_scalar('Test/mr', test_results['mr'], epoch)
        writer.add_scalar('Test/mrr', test_results['mrr'], epoch)
        writer.add_scalar('Test/hits_1', test_results[f'hits@{1}'], epoch)
        writer.add_scalar('Test/hits_3', test_results[f'hits@{3}'], epoch)
        writer.add_scalar('Test/hits_10', test_results[f'hits@{10}'], epoch)

        #if args.optimizer == 'ADAM':
            # lr = scheduler.get_last_lr()[-1]
        #    logging.info('[LR]\t%f', optimizer.param_groups[0]['lr'])

        #    if optimizer.param_groups[0]['lr'] >= args.learning_rate_min:
        #        scheduler.step(val_loss)
        #    else:
        #        print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD, KEEP MIN LR")
                # break

        if test_results[f'hits@{1}'] > best_hits_1:
            best_hits_1 = test_results[f'hits@{1}']
        if test_results[f'hits@{10}'] > best_hits_10:
            best_hits_10 = test_results[f'hits@{10}']
            best_epoch_hits = epoch
        if test_results['mrr'] > best_mrr:
            best_mrr = test_results['mrr']
            best_epoch_mrr = epoch
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, log_dir + '/model_best_mrr.pth')

        if epoch % 10 == 0:
            logging.info("Best_epoch_mrr {:04d} mrr {:.3f}".format(best_epoch_mrr, best_mrr))
            logging.info("Best_epoch_hits {:04d} hits_10 {:.3f}".format(best_epoch_hits, best_hits_10))

    logging.info('Duration is {} (day:h:min)'.format(calc_time(time.time() - start_time)))
    logging.info(
        "Best_epoch_mrr {:04d} | mrr {:.3f} | Best_epoch_hits {:04d} | hits_10 {:.3f}".format(best_epoch_mrr, best_mrr,
                                                                                              best_epoch_hits,
                                                                                              best_hits_10))
    logging.info('RUNDIR: {}'.format(log_dir))


def train(train_loader, g_train, model, optimizer, scheduler,epoch, criterion, save_freq, device):
    # lr = scheduler.get_last_lr()[-1]
    # logging.info('[LR]\t%f', lr)
    model.train()

    #epoch_loss = AverageMeter()
    train_loss = []

    #desc = '=> Train  '
    #t = tqdm(train_loader, desc=desc)
    for step, (triplets, labels) in enumerate(train_loader):
        start = time.time()
        triplets, labels = triplets.to(device), labels.to(device)
        subj, rel = triplets[:, 0], triplets[:, 1]

        pred = model(g_train, subj, rel)  # [batch_size, num_ent]
        #if step % 10==0:
        #    logging.info('GPU {} memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))
        #    gpu_memory_log()

        loss = criterion(pred, labels)

        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #n = pred.size(0)
        #epoch_loss.update(loss.item(), n)
        if step %10==0:
            # logging.info(
            #    "---Step {:04d} / Epoch {:04d}--- | ".format(step, epoch) + "Loss: {:.6f} | ".format(epoch_loss.avg) +"Time: {:.5f}| ".format(time.time() - start))
            logging.info(
                "---Step {:04d} / Epoch {:04d}--- | ".format(step, epoch) + "Loss: {:.6f} | ".format(loss.item()) +"Time: {:.5f}| ".format(time.time() - start))
        #t.set_postfix(time=time.time() - start, loss=epoch_loss.avg)

    train_loss = np.sum(train_loss)

    scheduler.step()

    stage = 'Train'
    logging.info(
        "{}_Epoch {:04d} | ".format(stage, epoch) +
        "{}_Loss: {:.3f} | ".format(stage, train_loss) +
        #"{}_Loss: {:.3f} | ".format(stage, epoch_loss.avg)+
        "{}_Time: {:.5f} | ".format(stage, time.time() - start))

    if epoch % save_freq == 0 and epoch > 1:
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, log_dir + '/model_' + str(epoch) + '_.pth')

    #return epoch_loss.avg
    return train_loss


def predict(val_test_loader, g, model, device):
    """
    infer only for predict "tail"
    Function to run model evaluation for a given mode
    :param split: valid or test, set which data-set to evaluate on
    :param mode: head or tail
    :return: results['mr']: Sum of ranks
             results['mrr']: Sum of Reciprocal Rank
             results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
             results['count']: number of total predictions
    """
    with torch.no_grad():
        results = dict()
        test_iter = val_test_loader
        model.eval()
        #test_loss = AverageMeter()
        test_loss = []
        for step, (triplets, labels) in enumerate(test_iter):
            triplets, labels = triplets.to(device), labels.to(device)
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = model(g, subj, rel)
            loss_test = F.binary_cross_entropy(pred, labels)
            test_loss.append(loss_test.item())
            if step % 10 == 0:
               logging.info(
                    "---Step {:04d}--- | ".format(step) + "Loss: {:.6f} | ".format(loss_test.item()))
            b_range = torch.arange(pred.shape[0], device=device)
            target_pred = pred[b_range, obj]  # [batch_size, 1], get the predictive score of obj
            # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred  # copy predictive score of obj to new pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]  # get the rank of each (sub, rel, obj)
            ranks = ranks.float()  # lower, better
            results['count'] = torch.numel(ranks) + results.get('count', 0)  # number of predictions
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0)
            for k in [1, 3, 10]:
                results[f'hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'hits@{k}', 0)

            #n = pred.size(0)
            #test_loss.update(loss_test.item(), n)
        test_loss = np.sum(test_loss)
    #return results, test_loss.avg
    return results, test_loss


def infer(model, g, tail_loader, head_loader, epoch, flag, device):
    """
    Function to evaluate the model on validation or test set
    :param split: valid or test, set which data-set to evaluate on
    :return: results['mr']: Average of ranks_left and ranks_right
             results['mrr']: Mean Reciprocal Rank
             results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
             results['left_mrr'], results['left_mr'], results['right_mrr'], results['right_mr']
             results['left_hits@k'], results['right_hits@k']
    """

    def get_combined_results(left, right):
        results = dict()
        assert left['count'] == right['count']
        count = float(left['count'])
        results['left_mr'] = round(left['mr'] / count, 5)
        results['left_mrr'] = round(left['mrr'] / count, 5)
        results['right_mr'] = round(right['mr'] / count, 5)
        results['right_mrr'] = round(right['mrr'] / count, 5)
        results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
        results['mrr'] = round((left['mrr'] + right['mrr']) / (2 * count), 5)
        for k in [1, 3, 10]:
            results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
            results[f'right_hits@{k}'] = round(right[f'hits@{k}'] / count, 5)
            results[f'hits@{k}'] = round((results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
        return results

    # model.eval()
    left_result, left_loss = predict(tail_loader, g, model, device)
    right_result, right_loss = predict(head_loader, g, model, device)
    results = get_combined_results(left_result, right_result)

    loss_test = 0.5 * (left_loss + right_loss)

    stage = flag
    logging.info(
        "{}_Epoch {:04d} | Loss {:04f}|".format(stage, epoch, loss_test) +
        "{}_MR: {:.3} | MRR: {:.3} | "
        "HITS@1: {:04f}| HITS@3: {:04f}| HITS@10: {:04f}|".format(stage, results['mr'], results['mrr'],
                                                                  results[f'hits@{1}'], results[f'hits@{3}'],
                                                                  results[f'hits@{10}']))
    return results, loss_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MG-NAS-LP")
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='Dataset to use, default: FB15k-237，wn18rr')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint to use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--init_fea_dim', type=int, default=128, help='number of initial embedding dim of features')
    parser.add_argument('--feature_dim', type=int, default=128, help='number of features')
    parser.add_argument('--num_base_r', type=int, default=23, help='relation-basis composition {475, 23}')
    parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-5, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=0e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer')
    parser.add_argument('--genotype', nargs='?', type=str,
                        default="[Genotype(alpha_cell=[('pre_mult', 1, 0), ('f_sparse_comp', 2, 1), ('f_sparse_comp', 3, 2), ('a_max', 4, 2), ('a_max', 5, 3), ('f_sparse_last', 6, 5), ('f_sparse_last', 7, 5)], concat_node=[4, 5, 6, 7], score_func='sf_ConvE')]")

    parser.add_argument('--gamma', type=float, default=40, help='margin for transE score func')
    parser.add_argument('--conve_hid_drop', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--k_w', dest='k_w', default=8, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=16, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=128, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--embed_dim', default=128, type=int, help='ConvE embedding dim')
    parser.add_argument('--ker_sz', dest='ker_sz', default=8, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('--save_model_freq', type=int, default=300, help='frequence for saving model')
    # parser.add_argument('--save_result', type=str, default="./save/Current_search.txt")
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for dataloader.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--dropout_cell', type=float, default=0.3, help='dropout out of cell for training')
    parser.add_argument('--drop_op', type=float, default=0, help='dropout out in each operation in cell for training')
    parser.add_argument('--drop_aggr', type=float, default=0.1, help='dropout out in aggregation func')

    args = parser.parse_args()

    log_dir = './' + args.save + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    writer = SummaryWriter(log_dir + '/tbx_log')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    # load_kg_wn18rr(args)
    start_batch(args)
