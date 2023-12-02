import os
import sys
import time
import random
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model.backbone.resnet import resnet101
from loss.asl import AsymmetricLoss

from utils.dataloader import get_data_loader
from utils.metrics import AverageMeter, compute_mAP, average_performance
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from config import arg_parse, logger, show_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Model(nn.Module):
    def __init__(self, classNum):
        super().__init__()
        self.classNum = classNum
        self.backbone = resnet101(pretrained=True, avg_pool=False)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, self.classNum)
    
    def forward(self, x):
        batchSize = x.shape[0]
        x = self.backbone(x)
        x = self.aap(x).view(batchSize, -1)
        x = self.fc(x)
        return x

def get_eta_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    eta = ''
    if d > 0:
        eta += '%dd,' % d

    eta += '%02d:%02d:%02d' % (h, m, s) 
    return eta

def main():
    setup_seed(1)

    # Argument Parse
    args = arg_parse('Base')

    os.makedirs('exp/log', exist_ok=True)
    os.makedirs('exp/summary', exist_ok=True)

    # Bulid Logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_path = 'exp/log/{}-{}.log'.format(args.post, time.strftime('%Y%m%d-%H%M%S'))
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Show Argument
    show_args(args)

    # Save Code File
    save_code_file(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader, test_loader = get_data_loader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    #GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.labels)

    model = Model(classNum=args.classNum)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cpu')
        args.startEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}".format(args.startEpoch))

    model.cuda()
    logger.info("==> Done!\n")

    criterion = {'ASL': AsymmetricLoss(gamma_neg=2)}

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.layer4.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weightDecay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepEpoch, gamma=0.1)

    if args.evaluate:
        Validate(test_loader, model, 0, args)
        return

    # logger.info('Total: {:.3f} GB'.format(torch.cuda.get_device_properties(0).total_memory/1024.0**3))

    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):

        Train(train_loader, model, criterion, optimizer, writer, epoch, args)
        save_checkpoint(args, {'epoch':epoch, 'state_dict':model.state_dict()}, 'last')

        scheduler.step()

    writer.close()
    Validate(test_loader, model, epoch, args)

def Train(train_loader, model, criterion, optimizer, writer, epoch, args):

    model.train()
    model.backbone.eval()
    model.backbone.layer4.train()

    loss = AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(train_loader):
        
        input, target = input.cuda(), target.float().cuda()

        # Log time of loading data
        data_time.update(time.time() - end)

        # Forward
        output = model(input)

        
        # Compute and log loss
        loss_ = criterion['ASL'](output, target)

        loss.update(loss_.item())

        # Backward
        loss_.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if (batchIndex + 1) % args.printFreq == 0:
            eta = batch_time.avg * (len(train_loader) - batchIndex - 1) + batch_time.avg * len(train_loader) * (args.epochs - epoch - 1)
            eta = get_eta_time(eta)

            logger.info('[Train] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'ETA {eta} Learn Rate {lr:g} Loss {loss.val:.3g} ({loss.avg:.3g})'.format(
                        epoch+1, batchIndex+1, len(train_loader), batch_time=batch_time, data_time=data_time,
                        eta=eta, lr=optimizer.param_groups[0]['lr'], loss=loss))
            sys.stdout.flush()

            global_step = epoch * len(train_loader) + batchIndex
            writer.add_scalar('Loss', loss.avg, global_step)


def Validate(val_loader, model, epoch, args):

    model.eval()

    preds, targets = [], [] 
    loss, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.float().cuda()
        
        # Log time of loading data
        data_time.update(time.time()-end)

        # Forward
        with torch.no_grad():
            output = model(input).sigmoid()

            # Compute loss and prediction
            loss_ = F.binary_cross_entropy(output, target)
        
        loss.update(loss_.item())

        # Change target to [0, 1]
        
        preds.append(output)
        targets.append(target)

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch        
        if (batchIndex + 1) % args.printFreq == 0:
            logger.info('[Test] [Epoch {0}]: [{1}/{2}] '
                        'Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'Loss {loss.val:.3g} ({loss.avg:.3g})'.format(
                        epoch+1, batchIndex+1, len(val_loader), 
                        batch_time=batch_time, data_time=data_time,
                        loss=loss))
            sys.stdout.flush()

    preds = torch.cat(preds, 0).cpu().numpy()
    targets = torch.cat(targets, 0).cpu().numpy()
    mAP, _ = compute_mAP(preds, targets)

    CP, CR, CF1, OP, OR, OF1 = average_performance(preds, targets, thr=0.5)
    # OP_K, OR_K, OF1_K, CP_K, CR_K, CF1_K = average_performance(preds, targets, k=3)

    logger.info('[Test] mAP: {mAP:.3f}, '
                'CP: {CP:.3f}, CR: {CR:.3f}, CF1:{CF1:.3f}, '
                'OP: {OP:.3f}, OR: {OR:.3f}, OF1: {OF1:.3f}'.format(
                mAP=mAP, OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

    return mAP


if __name__=="__main__":
    main()
