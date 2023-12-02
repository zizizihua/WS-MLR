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
from sklearn.cluster import KMeans

from model.SSGRL_Proto import SSGRL_Proto

from utils.dataloader import get_graph_and_word_file, get_data_loader, get_graph_file
from utils.metrics import AverageMeter, compute_mAP, average_performance
from utils.checkpoint import load_pretrained_model, save_code_file, save_checkpoint
from config import arg_parse, logger, show_args


queue_len = 128

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_eta_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    eta = ''
    if d > 0:
        eta += '%dd,' % d

    eta += '%02d:%02d:%02d' % (h, m, s) 
    return eta


def compute_prototype(model1, model2, train_loader1, train_loader2, args):
    model1.eval()
    model2.eval()

    num_batchs = min(len(train_loader1), len(train_loader2)) * 2
    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    batch_time, data_time = AverageMeter(), AverageMeter()
    logger.info("================ Compute Prototype ================")

    end = time.time()
    prototypes = []
    features = [torch.zeros(1, model1.low_dim) for i in range(args.classNum)]
    term = 0

    with torch.no_grad():
        for batchIndex in range(num_batchs):
            if term == 0:
                sampleIndex, input, target = next(iter1)
            else:
                sampleIndex, input, target = next(iter2)
        
            input, target = input.cuda(), target.float().cuda()

            # Log time of loading data
            data_time.update(time.time() - end)

            # Forward
            if term == 0:
                cls_logits1, semantic_feature1 = model1(input)
                cls_logits2, semantic_feature2 = model2(input)
            else:
                cls_logits1, semantic_feature1 = model2(input)
                cls_logits2, semantic_feature2 = model1(input)

            target = target.cpu()
            for feature in [semantic_feature1.cpu(), semantic_feature2.cpu()]:
                for i in range(args.classNum):
                    features[i] = torch.cat((features[i], feature[target[:, i] == 1, i]), dim=0)
        
            term = 1 - term

            batch_time.update(time.time() - end)
            end = time.time()

            if (batchIndex + 1) % args.printFreq == 0:
                eta = batch_time.avg * (num_batchs - batchIndex - 1)
                eta = get_eta_time(eta)

                logger.info('[Prototype] [{0}/{1}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} ETA {eta}'.format(
                            batchIndex+1, num_batchs, batch_time=batch_time, data_time=data_time, eta=eta))
                sys.stdout.flush()
    
        
        for i in range(args.classNum):
            kmeans = KMeans(n_clusters=1).fit(features[i].numpy())
            prototypes.append(torch.tensor(kmeans.cluster_centers_).cuda())

        prototypes = torch.cat(prototypes, dim=0)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        model1.prototypes = prototypes.clone()
        model2.prototypes = prototypes.clone()

    del features


def main():
    setup_seed(1)

    # Argument Parse
    args = arg_parse('DBMLCL')

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
    # save_code_file(args)

    # Create dataloader
    logger.info("==> Creating dataloader...")
    train_loader1, train_loader2, test_loader = get_data_loader(args)
    logger.info("==> Done!\n")

    # Load the network
    logger.info("==> Loading the network...")
    GraphFile, WordFile = get_graph_and_word_file(args, np.concatenate([train_loader1.dataset.labels, train_loader2.dataset.labels], axis=0))

    model1 = SSGRL_Proto(GraphFile, WordFile, classNum=args.classNum)
    model2 = SSGRL_Proto(GraphFile, WordFile, classNum=args.classNum)

    if args.pretrainedModel != 'None':
        logger.info("==> Loading pretrained model...")
        model1 = load_pretrained_model(model1, args)
        model2 = load_pretrained_model(model2, args)

    logger.info("==> Done!\n")

    for p in model1.backbone.parameters():
        p.requires_grad = False
    for p in model1.backbone.layer4.parameters():
        p.requires_grad = True
    for p in model2.backbone.parameters():
        p.requires_grad = False
    for p in model2.backbone.layer4.parameters():
        p.requires_grad = True

    optimizer1 = torch.optim.Adam(filter(lambda x: x.requires_grad, model1.parameters()), lr=args.lr, weight_decay=args.weightDecay)
    optimizer2 = torch.optim.Adam(filter(lambda x: x.requires_grad, model2.parameters()), lr=args.lr, weight_decay=args.weightDecay)

    scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.stepEpoch, gamma=0.1)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.stepEpoch, gamma=0.1)
    
    model1.cuda()
    model2.cuda()

    if args.resumeModel != 'None':
        logger.info("==> Loading checkpoint...")
        checkpoint = torch.load(args.resumeModel, map_location='cuda')
        args.startEpoch = checkpoint['epoch']+1
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        logger.info("==> Checkpoint Epoch: {0}".format(args.startEpoch))
    
    
    if args.evaluate:
        Validate(test_loader, model1, model2, args.startEpoch-1, args)
        return

    # logger.info('Total: {:.2f} GB'.format(torch.cuda.get_device_properties(0).total_memory/1024.0**3))

    # Running Experiment
    logger.info("Run Experiment...")
    writer = SummaryWriter('{}/{}'.format('exp/summary/', args.post))

    criterion = {'BCELoss': nn.BCEWithLogitsLoss().cuda(),
                 'CELoss': nn.CrossEntropyLoss().cuda()}

    for epoch in range(args.startEpoch, args.epochs):
        # if epoch >= 1:
        #     compute_prototype(model1, model2, train_loader1, train_loader2, args)
        Train(train_loader1, train_loader2, model1, model2, criterion, optimizer1, optimizer2, writer, epoch, args) #, predictions, similarities, labels)
        save_checkpoint(args, {'epoch':epoch, 'optimizer1': optimizer1.state_dict(), 'optimizer2': optimizer2.state_dict(), 
                               'model1':model1.state_dict(), 'model2':model2.state_dict()}, 
                               'before_clean' if epoch == args.cleanEpoch-2 else 'last')
        
        scheduler1.step()
        scheduler2.step()

    writer.close()
    mAP = Validate(test_loader, model1, model2, epoch, args)


def get_instance_logits(feature1, feature2, gt_labels, queue, temperature):
    q, k = feature1, feature2
    kn = queue
    classNum = q.shape[1]
    all_logits = []
    for i in range(classNum):
        pos_idx = (gt_labels[:, i] == 1).nonzero().view(-1)
        if len(pos_idx) == 0:
            continue

        l_pos = torch.einsum('bn,bn->b', [q[pos_idx, i], k[pos_idx, i]]).view(-1, 1)
        l_neg = torch.einsum('bn,nm->bm', [q[pos_idx, i], kn[:, i]])
        logits = torch.cat([l_pos, l_neg], dim=1)
        all_logits.append(logits)
    if len(all_logits) > 0:
        all_logits = torch.cat(all_logits, dim=0) / temperature
    else:
        all_logits = torch.zeros((0,))
    return all_logits


def get_proto_logits(feature, prototype, gt_labels, temperature):
    classNum = feature.shape[1]
    all_logits = []
    for i in range(classNum):
        pos_idx = (gt_labels[:, i] == 1).nonzero().view(-1)
        if len(pos_idx) == 0:
            continue

        l_pos = F.cosine_similarity(feature[pos_idx, i], prototype[pos_idx, i], dim=1).view(-1, 1)
        neg_cls = list(range(classNum))
        neg_cls.pop(i)
        l_neg = F.cosine_similarity(feature[pos_idx, i].unsqueeze(1).repeat(1, classNum-1, 1),
                                    prototype[pos_idx][:, neg_cls],
                                    dim=2)
        logits = torch.cat([l_pos, l_neg], dim=1)
        all_logits.append(logits)
    if len(all_logits) > 0:
        all_logits = torch.cat(all_logits, dim=0) / temperature
    else:
        all_logits = torch.zeros((0,))
    return all_logits


def Train(train_loader1, train_loader2, model1, model2, criterion, optimizer1, optimizer2, writer, epoch, args): #, predictions, similarities, labels):

    model1.train()
    model2.train()
    model1.backbone.eval()
    model1.backbone.layer4.train()
    model2.backbone.eval()
    model2.backbone.layer4.train()

    predictions = torch.zeros(queue_len, args.classNum, device='cuda')
    similarities = torch.zeros(queue_len, args.classNum, device='cuda')
    labels = torch.zeros(queue_len, args.classNum, device='cuda')
    q_idx = 0

    total_loss, cls_loss, inst_loss, proto_loss, dtl_loss1, dtl_loss2 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    term = 0
    num_batchs = min(len(train_loader1), len(train_loader2)) * 2
    iter1 = iter(train_loader1)
    iter2 = iter(train_loader2)
    logger.info("=========================================")

    end = time.time()

    for batchIndex in range(num_batchs):
        if args.warmupEpoch > 0 and epoch < args.warmupEpoch:
            ni = num_batchs * epoch + batchIndex
            nw = num_batchs * args.warmupEpoch
            xp = [0, nw-1]  # x interp
            for pg1, pg2 in zip(optimizer1.param_groups, optimizer2.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                pg1['lr'] = np.interp(ni, xp, [0., pg1['initial_lr']])
                pg2['lr'] = np.interp(ni, xp, [0., pg2['initial_lr']])

        if term == 0:
            sampleIndex, input, target = next(iter1)
        else:
            sampleIndex, input, target = next(iter2)
        
        input, target = input.cuda(), target.float().cuda()

        # Log time of loading data
        data_time.update(time.time() - end)
        bs = input.shape[0]

        # Forward
        if term == 0:
            model_ = model1
            opt = optimizer1
            cls_logits1, semantic_feature1 = model1(input)
            with torch.no_grad():
                cls_logits2, semantic_feature2 = model2(input)
                prototypes1 = model1.prototypes.clone().unsqueeze(0).repeat(bs, 1, 1)
                prototypes2 = model2.prototypes.clone().unsqueeze(0).repeat(bs, 1, 1)
        else:
            model_ = model2
            opt = optimizer2
            cls_logits1, semantic_feature1 = model2(input)
            with torch.no_grad():
                cls_logits2, semantic_feature2 = model1(input)
                prototypes1 = model2.prototypes.clone().unsqueeze(0).repeat(bs, 1, 1)
                prototypes2 = model1.prototypes.clone().unsqueeze(0).repeat(bs, 1, 1)
        
        with torch.no_grad():
            cls_score = (cls_logits1.sigmoid() + cls_logits2.sigmoid()) / 2
            similarity1 = F.cosine_similarity(semantic_feature1, prototypes1, dim=2)
            similarity2 = F.cosine_similarity(semantic_feature2, prototypes2, dim=2)
            similarity = (similarity1 + similarity2) / 2

            if (epoch + 1) >= args.cleanEpoch:
                theta1 = model1.theta1.clamp(0, 1)
                theta2 = model1.theta2.clamp(0, 1)
                alpha1 = model1.alpha1.clamp(-1, 1)
                alpha2 = model1.alpha2.clamp(-1, 1)

                noisy_mask = ((cls_score >= theta1) & (target == 0)) | ((cls_score < theta2) & (target == 1))
                proto_label = target.clone()
                proto_label[similarity < alpha2] = 0.
                proto_label[similarity >= alpha1] = 1.
                # alpha_ = 0.5 * alpha1 + 0.5 * alpha2
                # proto_label = (similarity >= alpha_).float()
                target[noisy_mask] = proto_label[noisy_mask]
            
            bs = cls_score.shape[0]
            predictions[q_idx:q_idx+bs] = cls_score
            similarities[q_idx:q_idx+bs] = similarity
            labels[q_idx:q_idx+bs] = target

            if q_idx + bs >= queue_len:
                model1.update_thresh(predictions, similarities, labels)

            q_idx = (q_idx + bs) % queue_len
        
        target_ = target.clone()
        feature_ = semantic_feature1.detach()
        model_.update_prototype(feature_, target_)

        # compute loss
        cls_loss_ = criterion['BCELoss'](cls_logits1, target)
        cls_loss.update(cls_loss_.item())

        inst_logits = get_instance_logits(semantic_feature1, semantic_feature2, target, model_.queue.clone(), model_.temperature)
        inst_loss_ = torch.zeros(1, device=target.device)
        if inst_logits.shape[0] > 0:
            inst_labels = torch.zeros(inst_logits.shape[0], dtype=torch.int64, device=inst_logits.device)
            inst_loss_ = criterion['CELoss'](inst_logits, inst_labels) * args.instLossWeight
        inst_loss.update(inst_loss_.item())

        loss = cls_loss_ + inst_loss_

        warmup_weight = batchIndex / num_batchs if epoch == 0 else 1
        proto_loss_ = torch.zeros(1, device=target.device)
        proto_logits = get_proto_logits(semantic_feature1, prototypes1, target, model_.temperature)
        if proto_logits.shape[0] > 0:
            # proto_labels = torch.zeros_like(proto_logits, requires_grad=False)
            # proto_labels[:, 0] = 1.
            # proto_loss_ = criterion['BCELoss'](proto_logits, proto_labels) * args.protoLossWeight * warmup_weight
            proto_labels = torch.zeros(proto_logits.shape[0], dtype=torch.int64, device=proto_logits.device)
            proto_loss_ = criterion['CELoss'](proto_logits, proto_labels) * args.protoLossWeight * warmup_weight
        loss = loss + proto_loss_
        proto_loss.update(proto_loss_.item())
        
        total_loss.update(loss.item())

        model_.dequeue_and_enqueue(semantic_feature2, target)
        
        # Backward
        loss.backward()
        opt.step()
        opt.zero_grad()

        term = 1 - term

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        if (batchIndex + 1) % args.printFreq == 0:
            eta = batch_time.avg * (num_batchs - batchIndex - 1) + batch_time.avg * num_batchs * (args.epochs - epoch - 1)
            eta = get_eta_time(eta)

            logger.info('[Train] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'ETA {eta} Learn Rate {lr:g} Cls Loss {cls_loss.val:.3g} ({cls_loss.avg:.3g}) Inst Loss {inst_loss.val:.3g} ' 
                        '({inst_loss.avg:.3g}) Proto Loss {proto_loss.val:.3g} ({proto_loss.avg:.3g}) '
                        'Total Loss {total_loss.val:.3g} ({total_loss.avg:.3g}) '
                        'theta1 {theta1:.3f} theta2 {theta2:.3f} alpha1 {alpha1:.3f} alpha2 {alpha2:.3f}'.format(
                        epoch+1, batchIndex+1, num_batchs, batch_time=batch_time, data_time=data_time, eta=eta, lr=opt.param_groups[0]['lr'], 
                        cls_loss=cls_loss, inst_loss=inst_loss, proto_loss=proto_loss, total_loss=total_loss, 
                        theta1=model1.theta1.mean(), theta2=model1.theta2.mean(), alpha1=model1.alpha1.mean(), alpha2=model1.alpha2.mean()))
            sys.stdout.flush()

            global_step = epoch * num_batchs + batchIndex
            writer.add_scalar('Cls Loss', cls_loss.avg, global_step)
            writer.add_scalar('Inst Loss', inst_loss.avg, global_step)
            writer.add_scalar('Proto Loss', proto_loss.avg, global_step)
    
    # if (epoch + 1) >= args.cleanEpoch:
    #     graph = get_graph_file(np.concatenate(labels, axis=0))
    #     model1.load_matrix(graph)
    #     model2.load_matrix(graph)

def Validate(val_loader, model1, model2, epoch, args):

    model1.eval()
    model2.eval()

    cls_pred, cls_pred1, cls_pred2, proto_pred, targets, new_targets = [], [], [], [], [], []
    cls_loss, batch_time, data_time = AverageMeter(), AverageMeter(), AverageMeter()
    pos_similarity, neg_similarity = AverageMeter(), AverageMeter()
    logger.info("=========================================")

    end = time.time()
    for batchIndex, (sampleIndex, input, target) in enumerate(val_loader):

        input, target = input.cuda(), target.float().cuda()
        
        # Log time of loading data
        data_time.update(time.time()-end)

        # Forward
        with torch.no_grad():
            cls_score1, proto_similarity1 = model1.test(input, return_proto=True)
            cls_score2, proto_similarity2 = model2.test(input, return_proto=True)
            cls_score = 0.5 * cls_score1 + 0.5 * cls_score2
            proto_similarity = 0.5 * proto_similarity1 + 0.5 * proto_similarity2
            proto_label = (proto_similarity >= 0.5 * model1.alpha1 + 0.5 * model1.alpha2).float()

            # Compute loss and prediction
            cls_loss_ = F.binary_cross_entropy(cls_score, target)
            pos_similarity.update(proto_similarity[target == 1].mean().item())
            neg_similarity.update(proto_similarity[target == 0].mean().item())

        cls_loss.update(cls_loss_.item())

        # Change target to [0, 1]
        cls_pred.append(cls_score)
        proto_pred.append(proto_label)
        cls_pred1.append(cls_score1)
        cls_pred2.append(cls_score2)
        targets.append(target)

        # Log time of batch
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info information of current batch        
        if (batchIndex + 1) % args.printFreq == 0:
            eta = batch_time.avg * (len(val_loader) - batchIndex - 1)
            eta = get_eta_time(eta)

            logger.info('[Test] [Epoch {0}]: [{1}/{2}] Batch Time {batch_time.avg:.3f} Data Time {data_time.avg:.3f} '
                        'ETA {eta} Cls Loss {cls_loss.val:.3g} ({cls_loss.avg:.3g}) '
                        'Pos Similarity {pos_similarity.val:.3g} ({pos_similarity.avg:.3g}) '
                        'Neg Similarity {neg_similarity.val:.3g} ({neg_similarity.avg:.3g})'.format(
                        epoch+1, batchIndex+1, len(val_loader), batch_time=batch_time, data_time=data_time, eta=eta, 
                        cls_loss=cls_loss, pos_similarity=pos_similarity, neg_similarity=neg_similarity))
            sys.stdout.flush()

    targets = torch.cat(targets, 0).cpu().numpy()
    pred_dict = {
        'cls': cls_pred,
        'cls1': cls_pred1,
        'cls2': cls_pred2,
        'proto': proto_pred
    }
    mAP = None
    for name, pred in pred_dict.items():
        pred = torch.cat(pred, 0).cpu().numpy()
        _mAP, _ = compute_mAP(pred, targets)
        CP, CR, CF1, OP, OR, OF1 = average_performance(pred, targets, thr=0.5)
        logger.info('[{name}] mAP: {mAP:.2f}, CP: {CP:.2f}, CR: {CR:.2f}, CF1: {CF1:.2f}, OP: {OP:.2f}, OR: {OR:.2f}, OF1: {OF1:.2f}'.format(
            name=name, mAP=_mAP, CP=CP, CR=CR, CF1=CF1, OP=OP, OR=OR, OF1=OF1))
        if name == 'cls':
            mAP = _mAP
    
    return mAP


if __name__=="__main__":
    main()
