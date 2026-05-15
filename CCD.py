import os
import sys
import time
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import clip
from model.CCD import ImageClassifier
from utils.ccd import build_text_weights, clip_softscore, cropping_box, get_class_names
from utils.dataloader import get_data_loader
from utils.metrics import AverageMeter, compute_mAP, average_performance
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from config import arg_parse, logger, show_args
from utils.device import resolve_device, setup_seed as setup_device_seed


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

    args = arg_parse('CCD')
    args.deviceObj = resolve_device(args)
    setup_device_seed(1, args.deviceObj)

    os.makedirs('exp/log', exist_ok=True)
    os.makedirs('exp/summary', exist_ok=True)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_path = 'exp/log/{}-{}.log'.format(args.post, time.strftime('%Y%m%d-%H%M%S'))
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    show_args(args)
    save_code_file(args)

    logger.info("==> Creating dataloader...")
    train_loader, test_loader = get_data_loader(args)
    logger.info("==> Done!\n")

    logger.info("==> Loading CLIP...")
    clip_model, _ = clip.load(args.clipModelPath, device=args.deviceObj)
    clip_model.eval()
    class_names = get_class_names(args.dataset)
    if len(class_names) != args.classNum:
        raise ValueError('Class name count {} does not match classNum {}'.format(len(class_names), args.classNum))
    text_weights = build_text_weights(clip, clip_model, class_names, args.deviceObj)
    logger.info("==> Done!\n")

    logger.info("==> Loading the network...")
    model = ImageClassifier(classNum=args.classNum,
                            pretrained=True,
                            freezeBackbone=args.freezeBackbone)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model = load_pretrained_model(model, args)

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location=args.deviceObj)
        args.startEpoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("==> Checkpoint Epoch: {0}".format(args.startEpoch))

    model.to(args.deviceObj)
    logger.info("==> Done!\n")

    optimizer = get_optimizer(model, args)

    if args.evaluate:
        Validate(test_loader, model, 0, args)
        return

    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    state = init_ccd_state(len(train_loader.dataset), args)
    best_mAP = 0

    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        state = Infer(train_loader, model, clip_model, text_weights, state, epoch, args)
        Train(train_loader, model, optimizer, writer, state, epoch, args)
        mAP, _ = Validate(test_loader, model, epoch, args)

        save_checkpoint(args, {'epoch': epoch, 'state_dict': model.state_dict()}, 'last')
        if mAP > best_mAP:
            best_mAP = mAP
            save_checkpoint(args, {'epoch': epoch, 'state_dict': model.state_dict()}, 'best')

    writer.close()


def get_optimizer(model, args):
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    head_params = [param for param in model.onebyone_conv.parameters() if param.requires_grad]
    opt_params = [
        {'params': backbone_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lrMult * args.lr}
    ]
    return torch.optim.Adam(opt_params, lr=args.lr, weight_decay=args.weightDecay)


def init_ccd_state(data_size, args):
    return {
        'pseudo': torch.zeros(data_size, args.classNum).to(args.deviceObj),
        'flag': torch.ones(data_size, 1).to(args.deviceObj),
        'initialized': False,
        'clipBias': torch.ones(args.classNum).to(args.deviceObj),
        'multiplier': 0.05 if args.LS else 0.001
    }


def calibrate_multiplier(train_loader, model, multiplier, args):
    class_ones = torch.ones(args.classNum).to(args.deviceObj)
    class_zeros = torch.zeros(args.classNum).to(args.deviceObj)
    avg_count = 0
    max_count = 0
    max_index = 0

    model.eval()
    for batchIndex, (sampleIndex, input, target) in enumerate(train_loader):
        input = input.to(args.deviceObj)

        with torch.no_grad():
            logits, _ = model(input, True)
            preds = torch.sigmoid(logits)
            if preds.dim() == 1:
                preds = preds.unsqueeze(0)

        class_above_thres = torch.where(preds >= multiplier, class_ones, class_zeros)
        cls_count = class_above_thres.sum(dim=1)
        avg_count += cls_count.sum().item() / 100
        batch_max, batch_argmax = cls_count.max(dim=0)
        if batch_max.item() > max_count:
            max_count = batch_max.item()
            max_index = sampleIndex[batch_argmax].item()

        if batchIndex % 100 == 0 and batchIndex != 0:
            logger.info('max above thres is {0} and sample index is {1} \n avg above thres is {2}'.format(
                max_count, max_index, avg_count))
            if args.bound < avg_count < args.bound + 2:
                logger.info('fixed multiplier is {0}'.format(multiplier))
                return multiplier
            multiplier += 0.001
            avg_count = 0
            max_count = 0

    return multiplier


def Infer(train_loader, model, clip_model, text_weights, state, epoch, args):
    model.eval()
    classwise_logit = torch.zeros(args.classNum).to(args.deviceObj)
    temp_counts = torch.zeros(args.classNum).to(args.deviceObj)
    class_ones = torch.ones(args.classNum).to(args.deviceObj)
    class_zeros = torch.zeros(args.classNum).to(args.deviceObj)
    ls_coeff = 1 / args.LSCoeff
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")
    logger.info("[Infer] [Epoch {0}]: updating CCD pseudo labels".format(epoch + 1))

    if epoch == 2:
        state['multiplier'] = calibrate_multiplier(train_loader, model, state['multiplier'], args)

    if epoch >= 2 + args.infNum:
        return state

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(train_loader):
        input = input.to(args.deviceObj)
        target = target.float().to(args.deviceObj)
        height, width = input.shape[-2], input.shape[-1]
        data_time.update(time.time() - end)

        with torch.no_grad():
            if epoch == 0:
                softscore = clip_softscore(clip_model, input, text_weights, height, width)
                entropy = -torch.sum(softscore * torch.log(softscore + 1e-10), dim=1)
                _, topk_indices = softscore.topk(1)
                topk_flag = F.one_hot(topk_indices, num_classes=args.classNum).sum(dim=1).float()
                clean_mask = entropy <= 2

                state['flag'][sampleIndex] = clean_mask.float().unsqueeze(1)
                if clean_mask.any():
                    temp_counts += topk_flag[clean_mask].sum(dim=0)
                    classwise_logit += (softscore[clean_mask] * topk_flag[clean_mask]).sum(dim=0)

                state['pseudo'][sampleIndex] = softscore
            elif args.updateLabel and epoch > 1:
                logits, CAM = model(input, True)
                preds = torch.sigmoid(logits)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(0)

                for imageIndex in range(input.size(0)):
                    table_index = sampleIndex[imageIndex].item()
                    class_above_thres = torch.where(preds[imageIndex] >= state['multiplier'], class_ones, class_zeros)
                    cls_indices = class_above_thres.nonzero()
                    if cls_indices.numel() == 0:
                        continue

                    crop_scores = []
                    for cls_idx in cls_indices:
                        temp_cam = CAM[imageIndex:imageIndex + 1, cls_idx.item()]
                        temp_cam = F.interpolate(temp_cam.unsqueeze(0), (height, width), mode='bilinear', align_corners=False)
                        cam_min = temp_cam.amin(dim=(2, 3), keepdim=True)
                        cam_max = temp_cam.amax(dim=(2, 3), keepdim=True)
                        temp_cam_norm = (temp_cam - cam_min) / (cam_max - cam_min + 1e-5)
                        temp_above = (temp_cam_norm > 0.95).nonzero()
                        if temp_above.numel() == 0:
                            continue

                        _, _, box1_h, box1_w = temp_above[0]
                        _, _, box2_h, box2_w = temp_above[-1]
                        crop_clip, _ = cropping_box(input[imageIndex:imageIndex + 1], (box1_h, box1_w), (box2_h, box2_w), args.offsetSize)
                        crop_softscore = clip_softscore(clip_model, crop_clip, text_weights, crop_clip.shape[-2], crop_clip.shape[-1])
                        crop_scores.append(crop_softscore)

                    if len(crop_scores) == 0:
                        continue

                    temp_label = torch.cat(crop_scores, dim=0).amax(dim=0, keepdim=True)
                    if args.LS:
                        temp_label = torch.where(temp_label >= ls_coeff, temp_label, ls_coeff * class_ones.unsqueeze(0))
                    if args.localTemp:
                        temp_label = temp_label / state['clipBias'].unsqueeze(0)

                    state['pseudo'][table_index] = (
                        args.ratio * state['pseudo'][table_index] + (1 - args.ratio) * temp_label.squeeze(0)
                    ).clamp(0, 1)
        
        batch_time.update(time.time() - end)
        end = time.time()

        if (batchIndex + 1) % args.printFreq == 0:
            eta = batch_time.avg * (len(train_loader) - batchIndex - 1)
            eta = get_eta_time(eta)
            logger.info('[Infer] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'ETA {eta}'.format(epoch + 1, batchIndex + 1, len(train_loader),
                                           batch_time=batch_time, data_time=data_time, eta=eta))
            sys.stdout.flush()

    if epoch == 0:
        clip_bias = classwise_logit / temp_counts.clamp(min=1e-6)
        state['clipBias'] = clip_bias.clamp(min=1e-6)
        if args.LS:
            state['pseudo'] = torch.where(state['pseudo'] >= ls_coeff, state['pseudo'], ls_coeff * class_ones.unsqueeze(0))
        if args.globalTemp:
            state['pseudo'] = (state['pseudo'] / state['clipBias'].unsqueeze(0)).clamp(0, 1)
        if args.coeff > 0:
            model.alpha.mul_(args.coeff)
        state['initialized'] = True

    return state


def Train(train_loader, model, optimizer, writer, state, epoch, args):
    model.train()

    loss_meter = AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(train_loader):
        input = input.to(args.deviceObj)

        data_time.update(time.time() - end)

        batch_label = state['pseudo'][sampleIndex].float()
        batch_flag = state['flag'][sampleIndex].float()

        logits = model(input)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        loss_ce = F.binary_cross_entropy_with_logits(logits, batch_label, reduction='none')
        if epoch >= 2 and args.useConsist > 0:
            batch_flag = (batch_flag + args.lossCoeff).clamp(min=0, max=1)
            loss = (batch_flag * loss_ce).mean()
        else:
            loss = loss_ce.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if (batchIndex + 1) % args.printFreq == 0:
            eta = batch_time.avg * (len(train_loader) - batchIndex - 1) + batch_time.avg * len(train_loader) * (args.epochs - epoch - 1)
            eta = get_eta_time(eta)
            logger.info('[Train] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'ETA {eta} Learn Rate {lr:g} Loss {loss.val:.3g} ({loss.avg:.3g})'.format(
                        epoch + 1, batchIndex + 1, len(train_loader), batch_time=batch_time, data_time=data_time,
                        eta=eta, lr=optimizer.param_groups[0]['lr'], loss=loss_meter))
            sys.stdout.flush()

            global_step = epoch * len(train_loader) + batchIndex
            writer.add_scalar('Loss', loss_meter.avg, global_step)


def Validate(val_loader, model, epoch, args):
    model.eval()

    preds, targets = [], []
    loss, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(val_loader):
        input, target = input.to(args.deviceObj), target.float().to(args.deviceObj)

        data_time.update(time.time() - end)

        with torch.no_grad():
            output = model(input).sigmoid()
            loss_ = F.binary_cross_entropy(output, target)

        loss.update(loss_.item())
        preds.append(output)
        targets.append(target)

        batch_time.update(time.time() - end)
        end = time.time()

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    mAP, APs = compute_mAP(preds, targets)
    CP, CR, CF1, OP, OR, OF1 = average_performance(preds.cpu().numpy(), targets.cpu().numpy(), k=3)

    logger.info('[Test] [Epoch {0}]: Loss {loss.avg:.3g} mAP {mAP:.3f}'.format(epoch + 1, loss=loss, mAP=mAP))
    logger.info('CP {0:.2f} CR {1:.2f} CF1 {2:.2f} OP {3:.2f} OR {4:.2f} OF1 {5:.2f}'.format(
        CP, CR, CF1, OP, OR, OF1))
    logger.info(APs)
    sys.stdout.flush()

    return mAP, APs


if __name__ == '__main__':
    main()
