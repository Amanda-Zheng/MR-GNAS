# This is for node classification for datasets AIFB, MUTAG, BGS, and AM.
# author: Amanda
# date: 2022/11
import datetime
import sys
import os
import time
import torch
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter

sys.path.append('..')

from compGCN.utils import *
from tqdm import tqdm
from models.model import Network
import torch.backends.cudnn as cudnn


def start_batch(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        #sys.exit(1)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    max_patience = args.max_patience
    cudnn.benchmark = True
    cudnn.enabled = True
    device = torch.device(args.device)
    start_time = time.time()
    logging.info('RUNDIR: {}'.format(log_dir))
    logging.info("args = %s", args)

    triple_g, g, train_idx, target_idx, val_idx, test_idx, labels, num_rels, num_classes, inv_target = load_batch(args,
                                                                                                                  device)
    print(test_idx)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.layers, return_eids=True)

    loader = dgl.dataloading.NodeDataLoader(
        g,
        target_idx[train_idx],
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # # validation sampler
    # val_loader = dgl.dataloading.NodeDataLoader(
    #     g,
    #     target_idx[val_idx],
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=args.num_workers)

    # test sampler
    # test_sampler = dgl.dataloading.MultiLayerNeighborSampler([None])
    test_loader = dgl.dataloading.NodeDataLoader(
        g,
        target_idx[test_idx],
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    if len(args.genotype) > 0:
        genotype = eval(args.genotype)
        logging.info('=> loading genotype: {}'.format(genotype))
    else:
        logging.info("Unknown genotype.")
        exit()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = WeightedCE(num_classes, device)
    criterion = criterion.to(device)

    # Model definition
    model = Network(device, genotype, g.number_of_nodes(), num_classes, num_rels, args.layers, args.zero_nodes, args.nodes,
                    args.feature_dim, args.init_fea_dim, args.num_base_r, criterion, args)

    model = model.to(device)
    model.apply(weights_init)

    logging.info("param size = %fMB", count_parameters_in_MB(model))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
        #                                                        eta_min=args.learning_rate_min)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=float(args.epochs), gamma=1)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5,
                                                               patience=5,
                                                               verbose=True)


    best_micro_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        logging.info('[EPOCH]\t%d', epoch)

        macro_acc, micro_acc, train_obj = train(triple_g, loader, labels, inv_target, model, optimizer, scheduler, epoch, criterion)
        writer.add_scalar('Train/macro_acc', macro_acc, epoch)
        writer.add_scalar('Train/micro_acc', micro_acc, epoch)
        writer.add_scalar('Train/loss', train_obj, epoch)
        # validation
        # macro_acc, micro_acc, valid_obj = infer(triple_g, val_loader, labels, inv_target, model, epoch, 'Val', criterion)
        # writer.add_scalar('Val/macro_acc', macro_acc, epoch)
        # writer.add_scalar('Val/micro_acc', micro_acc, epoch)
        # writer.add_scalar('Val/loss', valid_obj, epoch)
        # testing
        macro_acc, micro_acc, test_obj = infer(triple_g, test_loader, labels, inv_target, model, epoch, 'Test', criterion)
        writer.add_scalar('Test/macro_acc', macro_acc, epoch)
        writer.add_scalar('Test/micro_acc', micro_acc, epoch)
        writer.add_scalar('Test/loss', test_obj, epoch)
        if micro_acc > best_micro_acc:
            best_micro_acc = micro_acc
            best_epoch = epoch
        if epoch % 10 == 0:
            logging.info("Best_Epoch {:04d} acc {:.3f} | ".format(best_epoch, best_micro_acc))

    logging.info('Duration is {} (day:h:min)'.format(calc_time(time.time() - start_time)))
    logging.info("Best_Epoch {:04d} acc {:.3f} | ".format(best_epoch, best_micro_acc))
    logging.info('RUNDIR: {}'.format(log_dir))


def train(triple_g, loader, labels, inv_target, model, optimizer, scheduler, epoch, criterion):
    lr = scheduler.get_last_lr()[-1]
    logging.info('[LR]\t%f', lr)

    model.train()
    epoch_loss = AverageMeter()
    micro_acc = AverageMeter()
    macro_acc = AverageMeter()
    desc = '=> Train  '
    t = tqdm(loader, desc=desc)
    for step, sample_data in enumerate(t):
        start = time.time()
        input_nodes, seeds, block = sample_data
        seeds = inv_target[seeds]

        optimizer.zero_grad()
        logits = model(triple_g, block)

        if step == 0:
            logging.info('GPU {} memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

        loss = criterion(logits, labels[seeds])
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)

        n = logits.size(0)
        epoch_loss.update(loss.item(), n)
        micro_acc.update(train_acc, n)
        macro_acc.update(train_acc, n)
        t.set_postfix(time=time.time() - start, loss=epoch_loss.avg,
                      MACRO_ACC=macro_acc.avg, MICRO_ACC=micro_acc.avg)

    scheduler.step()

    stage = 'Train'
    logging.info(
        "{}_Epoch {:04d} | ".format(stage, epoch) +
        "{}_MacroAcc: {:.3f} | {}_MicroAcc: {:.3f} | {}_Loss: {:.3f} | ".format(stage, macro_acc.avg, stage, micro_acc.avg, stage, epoch_loss.avg))
    return macro_acc.avg, micro_acc.avg, epoch_loss.avg


def infer(triple_g, val_loader, labels, inv_target, model, epoch, stage, criterion):
    model.eval()
    epoch_loss = AverageMeter()
    micro_acc = AverageMeter()
    macro_acc = AverageMeter()
    desc = f'=> {stage}'
    t = tqdm(val_loader, desc=desc)
    with torch.no_grad():
        for step, val_sample_data in enumerate(val_loader):
            start = time.time()
            _, v_seeds, v_block = val_sample_data
            # map the seed nodes back to their type-specific ids, so that they
            # can be used to look up their respective labels
            v_seeds = inv_target[v_seeds]
            logits = model(triple_g, v_block)
            v_loss = criterion(logits, labels[v_seeds])
            v_acc = torch.sum(logits.argmax(dim=1) == labels[v_seeds]).item() / len(v_seeds)

            n = logits.size(0)
            micro_acc.update(v_acc, n)
            epoch_loss.update(v_loss.item(), n)
            macro_acc.update(v_acc, n)
            t.set_postfix(time=time.time() - start, loss=epoch_loss.avg,
                          MACRO_ACC=macro_acc.avg, MICRO_ACC=micro_acc.avg)

    logging.info(
        "{}_Epoch {:04d} | ".format(stage, epoch) +
        "{}_MacroAcc: {:.3f} | {}_MicroAcc: {:.3f} | {}_Loss: {:.3f} | ".format(stage, macro_acc.avg, stage, micro_acc.avg, stage, epoch_loss.avg))
    return macro_acc.avg, micro_acc.avg, epoch_loss.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MG-NAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # data
    parser.add_argument('--dataset', type=str, default='aifb', help='data name={mutag, am, aifb, bgs}')
    parser.add_argument('--data_type', type=str, default='nc', help='data type')
    parser.add_argument('--data_name', type=str, default='SBM_PATTERN', help='data name')
    # model
    parser.add_argument('--genotype', nargs='?', type=str, default="[Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_dense', 2, 1), ('f_sparse', 3, 2), ('f_identity', 4, 3), ('a_sum', 5, 2), ('a_sum', 6, 3), ('a_mean', 7, 4), ('f_dense_last', 8, 7), ('f_sparse_last', 9, 7), ('f_sparse_last', 10, 5)], concat_node=[5, 6, 7, 8, 9, 10]), Genotype(alpha_cell=[('pre_sub', 1, 0), ('f_sparse', 2, 1), ('f_identity', 3, 2), ('f_identity', 4, 1), ('a_max', 5, 2), ('a_mean', 6, 3), ('a_mean', 7, 4), ('f_sparse_last', 8, 7), ('f_sparse_last', 9, 8), ('f_identity', 10, 9)], concat_node=[5, 6, 7, 8, 9, 10])]", help='Model architecture')
    parser.add_argument('--readout', type=str, default='mean', help='graph read out')
    parser.add_argument('--layers', type=int, default=2, help='total number of layers')
    parser.add_argument('--init_fea_dim', type=int, default=16, help='number of initial embedding dim of features')
    parser.add_argument('--feature_dim', type=int, default=64, help='number of features')
    parser.add_argument('--num_base_r', type=int, default=50, help='relation-basis composition')
    parser.add_argument('--zero_nodes', type=int, default=1, help='zero nodes with edges information')
    parser.add_argument('--nodes', type=int, default=3, help='total number of nodes')
    parser.add_argument('--op_norm', action='store_true', default=False)
    # train
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=0e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    # save and report
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--max_patience', type=int, default=20, help='Max patience for early stopping')
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cpu')
    parser.add_argument('--valid', type=bool, default=False, help='whether split train/valid/test')

    args = parser.parse_args()

    log_dir = './' + args.save + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    writer = SummaryWriter(log_dir + '/tbx_log')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start_batch(args)
