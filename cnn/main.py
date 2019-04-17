import argparse
import os
import random
import shutil
import time
import glob
import warnings
import sys
import utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model_search import Network
from architect import Architect

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar-10 Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')#DARTS
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
#parser.add_argument('-b', '--batch-size', default=256, type=int,
#                    metavar='N',
#                    help='mini-batch size (default: 256), this is the total '
#                         'batch size of all GPUs on the current node when '
#                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')#DARTS
#parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
#                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')#DARTS
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')#DARTS
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
#parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                    metavar='W', help='weight decay (default: 1e-4)',
#                    dest='weight_decay')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')#DARTS
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=2, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')#DARTS
parser.add_argument('--layers', type=int, default=8, help='total number of layers')#DARTS
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')#DARTS
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')#DARTS
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')#DARTS
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')#DARTS
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')#DARTS
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')#DARTS
parser.add_argument('--save', type=str, default='EXP', help='experiment name')#DARTS
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')#DARTS
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')#DARTS
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')#DARTS
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')#DARTS

CIFAR_CLASSES = 10

best_acc1 = 0


class MyDataParallel(torch.nn.DataParallel):
    def arch_parameters(self):
        return self.module.arch_parameters()

    #def _loss(self, input, target):
    #    return self.module._loss(input, target)
    @property
    def _criterion(self):
        return self.module._criterion

    def new(self):
        return self.module.new()

    def genotype(self):
        return self.module.genotype()

    @property
    def alphas_normal(self):
        return self.module.alphas_normal

    @property
    def alphas_reduce(self):
        return self.module.alphas_reduce


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def arch_parameters(self):
        return self.module.arch_parameters()

    #def _loss(self, input, target):
    #    return self.module._loss(input, target)
    @property
    def _criterion(self):
        return self.module._criterion

    def new(self):
        return self.module.new()

    def genotype(self):
        return self.module.genotype()

    @property
    def alphas_normal(self):
        return self.module.alphas_normal

    @property
    def alphas_reduce(self):
        return self.module.alphas_reduce


def main():
    """
    single-gpu: python main.py --gpu 1 (suppose you want to use cuda:1)
    single-process multi-gpu: python main.py
    multi-process multi-gpu (both single-node and multi-node): e.g., python main.py --dist-url 'tcp://127.0.0.1:6666' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
    """
    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        #cudnn.benchmark = True#DARTS
        torch.manual_seed(args.seed)
        cudnn.enabled = True#DARTS
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    #global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        #print("=> creating model '{}'".format(args.arch))
        print("=> creating model")
        #model = models.__dict__[args.arch]()
        # (jones.wz) TO DO: support distributed cases.
        #torch.cuda.set_device(args.gpu)#DARTS
        #criterion = nn.CrossEntropyLoss()#DARTS
        #criterion = criterion.cuda(args.gpu)#DARTS
        #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)#DARTS
        #model.cuda()#DARTS

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            criterion = nn.CrossEntropyLoss()#DARTS
            criterion = criterion.cuda(args.gpu)#DARTS
            model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)#DARTS
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model = MyDistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        criterion = nn.CrossEntropyLoss()#DARTS
        criterion = criterion.cuda(args.gpu)#DARTS
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)#DARTS
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #    model.features = torch.nn.DataParallel(model.features)
        #    model.cuda()
        #else:
        #    model = torch.nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        #model = torch.nn.DataParallel(model).cuda()#DARTS
        model = MyDataParallel(model).cuda()#DARTS

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transforms.Compose([
    #        transforms.RandomResizedCrop(224),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ]))
    train_transform, valid_transform = utils._data_transforms_cifar10(args)#DARTS
    considered_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)#DARTS
    num_examples = len(considered_dataset)#DARTS
    indices = list(range(num_examples))#DARTS
    split = int(np.floor(args.train_portion * num_examples))#DARTS
    train_dataset = torch.utils.data.Subset(considered_dataset, indices[:split])
    valid_dataset = torch.utils.data.Subset(considered_dataset, indices[split:])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)#DARTS
    else:
        #train_sampler = None
        #train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])#DARTS
        #valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])#DARTS
        train_sampler = None
        valid_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    #val_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(valdir, transforms.Compose([
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)
    #DARTS:
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

    #if args.evaluate:
    #    validate(val_loader, model, criterion, args)
    #    return

    #DARTS:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    architect = Architect(model, args)

    #for epoch in range(args.start_epoch, args.epochs):
    for epoch in range(args.epochs):#DARTS
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)
        lr = scheduler.get_lr()[0]#DARTS
        print("epoch %d lr %e"%(epoch, lr))#DARTS

        genotype = model.genotype()#DARTS
        print('genotype = {}'.format(genotype))#DARTS

        print(F.softmax(model.alphas_normal, dim=-1))#DARTS
        print(F.softmax(model.alphas_reduce, dim=-1))#DARTS

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, args)
        train_acc, train_obj = train(train_loader, val_loader, model, architect, criterion, optimizer, lr, args)#DARTS
        print("train_acc %f"%(train_acc))#DARTS

        # evaluate on validation set
        #acc1 = validate(val_loader, model, criterion, args)
        valid_acc, valid_obj = infer(val_loader, model, criterion, args)#DARTS
        print('valid_acc %f'%(valid_acc))#DARTS

        # remember best acc@1 and save checkpoint
        #is_best = acc1 > best_acc1
        #best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            utils.save(model, os.path.join(args.save, 'weights.pt'))#DARTS
        #    save_checkpoint({
        #        'epoch': epoch + 1,
        #        'arch': args.arch,
        #        'state_dict': model.state_dict(),
        #        'best_acc1': best_acc1,
        #        'optimizer' : optimizer.state_dict(),
        #    }, is_best)


#DARTS:
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        
        #input = Variable(input, requires_grad=False).cuda()
        #target = Variable(target, requires_grad=False).cuda(async=True)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)


        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        #input_search = Variable(input_search, requires_grad=False).cuda()
        #target_search = Variable(target_search, requires_grad=False).cuda(async=True)
        if args.gpu is not None:
            input_search = input_search.cuda(args.gpu, non_blocking=True)
        target_search = target_search.cuda(args.gpu, non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            #logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            print('train %03d %e %f %f'%(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


"""
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
"""


#DARTS:
def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        #input = Variable(input, volatile=True).cuda()
        #target = Variable(target, volatile=True).cuda(async=True)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            #logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            print('valid %03d %e %f %f'%(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


"""
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
"""


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
