# This is for node classification for datasets AIFB, MUTAG, BGS, and AM.
# author: Amanda
# date: 2021/09/27


import argparse
import datetime
import logging
import sys
import time

import torch.backends.cudnn as cudnn
import torch.nn
from tensorboardX import SummaryWriter

sys.path.append('..')

from utils.utils import *
from models.model_search import Network
from models.architect import Architect
import numpy as np


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

    # fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    triple_g, g, train_idx, target_idx, val_idx, test_idx, labels, num_rels, num_classes, inv_target = load_batch(args,
                                                                                                                  device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.layers, return_eids=True)
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

    loader = dgl.dataloading.NodeDataLoader(
        g,
        target_idx[train_idx],
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # validation sampler
    val_loader = dgl.dataloading.NodeDataLoader(
        g,
        target_idx[val_idx],
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

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

    # Model definition
    model = Network(device, g.number_of_nodes(), num_classes, num_rels, args.layers, args.zero_nodes, args.nodes,
                    args.feature_dim, args.init_fea_dim, args.num_base_r)

    model = model.to(device)
    model.apply(weights_init)
    best_geno = str(model.show_genotypes())

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = WeightedCE(num_classes, device)
    criterion = criterion.to(device)

    # Calculating model's parameters
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    # Define optimizer for model updating & lr deccay for better learning

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=float(args.epochs), gamma=1)

    # Define searched architecture
    architect = Architect(device, model, args)

    for epoch in range(args.epochs):

        if epoch % args.save_freq == 0:
            logging.info(model.show_genotypes())
            for i in range(args.layers):
                logging.info('layer = %d', i)
                genotype = model.show_genotype(i)
                logging.info('genotype = %s', genotype)
            '''
            w1, w2, w3 = model.show_weights(0)
            print('[1] weights in first cell\n',w1)
            print('[2] weights in middle cell\n', w2)
            print('[3] weights in last cell\n', w3)
            '''
        # training
        # print('begin to train...')
        train_acc, train_loss, archi_loss = train(triple_g, loader, val_loader, labels, inv_target, model, architect,
                                                  optimizer, scheduler, epoch, warm_epochs, criterion)

        writer.add_scalar('Train/train_acc', train_acc, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Arch/loss', archi_loss, epoch)

        # true validation
        val_acc, val_loss = infer(triple_g, val_loader, labels, inv_target, model, epoch, 'Val', criterion)
        writer.add_scalar('Val/val_acc', val_acc, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)

        # testing
        test_acc, test_loss = infer(triple_g, test_loader, labels, inv_target, model, epoch, 'Test', criterion)
        writer.add_scalar('Test/test_acc', test_acc, epoch)
        writer.add_scalar('Test/loss', test_loss, epoch)

        if epoch > warm_epochs:
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


def train(triple_g, loader, val_loader, labels, inv_target, model, architect, optimizer, scheduler, epoch, warm_epochs,
          criterion):
    lr = scheduler.get_last_lr()[-1]
    logging.info('[LR]\t%f', lr)

    model.train()
    epoch_loss = AverageMeter()
    all_train_acc = AverageMeter()
    all_arch_loss = AverageMeter()

    for step, sample_data in enumerate(loader):

        input_nodes, seeds, block = sample_data
        seeds = inv_target[seeds]

        _, val_seeds, val_block = next(iter(val_loader))
        val_seeds = inv_target[val_seeds]

        if epoch > warm_epochs:
            architect.step(triple_g, block, labels, seeds, val_block, val_seeds, lr, optimizer, unrolled=args.unrolled)

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
        all_train_acc.update(train_acc, n)
        all_arch_loss.update(architect.loss.item(), n)

    scheduler.step()

    stage = 'Train'
    logging.info(
        "{}_Epoch {:04d} | ".format(stage, epoch) +
        "{}_Acc: {:.3f} | {}_Loss: {:.3f} | Arch_Loss: {:.3f}".format(stage, all_train_acc.avg, stage, epoch_loss.avg,
                                                                      all_arch_loss.avg))

    return all_train_acc.avg, epoch_loss.avg, all_arch_loss.avg


def infer(triple_g, val_loader, labels, inv_target, model, epoch, stage, criterion):
    model.eval()
    epoch_loss = AverageMeter()
    all_val_acc = AverageMeter()
    with torch.no_grad():
        for step, val_sample_data in enumerate(val_loader):
            _, v_seeds, v_block = val_sample_data
            # map the seed nodes back to their type-specific ids, so that they
            # can be used to look up their respective labels
            v_seeds = inv_target[v_seeds]
            logits = model(triple_g, v_block)
            v_loss = criterion(logits, labels[v_seeds])
            v_acc = torch.sum(logits.argmax(dim=1) == labels[v_seeds]).item() / len(v_seeds)

            n = logits.size(0)
            epoch_loss.update(v_loss.item(), n)
            all_val_acc.update(v_acc, n)

    logging.info(
        "{}_Epoch {:04d} | ".format(stage, epoch) +
        "{}_Acc: {:.3f} | {}_Loss: {:.3f} | ".format(stage, all_val_acc.avg, stage, epoch_loss.avg))

    return all_val_acc.avg, epoch_loss.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MG-NAS")
    parser.add_argument('--dataset', type=str, default='aifb', help='data name={mutag, am, aifb, bgs}')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument("--fanout", type=str, default="4, 4", help="Fan-out of neighbor sampling.")
    # parser.add_argument("--fanout", type=str, default="4", help="Fan-out of neighbor sampling.")
    parser.add_argument('--layers', type=int, default=2, help='total number of layers')
    parser.add_argument('--init_fea_dim', type=int, default=16, help='number of initial embedding dim of features')
    parser.add_argument('--feature_dim', type=int, default=64, help='number of features')
    parser.add_argument('--num_base_r', type=int, default=50, help='relation-basis composition')
    parser.add_argument('--zero_nodes', type=int, default=1, help='zero nodes with edges information')
    parser.add_argument('--nodes', type=int, default=3, help='total number of nodes')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=0e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_result', type=str, default="./save/Current_search.txt")
    # parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--max_patience', type=int, default=20, help='Max patience for early stopping')
    parser.add_argument('--warm_epochs', type=int, default=20, help='Warm up epochs to freeze architecture params')
    parser.add_argument('--valid', type=bool, default=True, help='whether split train/valid/test')

    args = parser.parse_args()

    log_dir = './' + args.save + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    writer = SummaryWriter(log_dir + '/tbx_log')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'search.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start_batch(args)
