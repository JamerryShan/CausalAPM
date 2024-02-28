from distutils.log import set_threshold
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args, tensor_writer=None, mi_estimator=None, mi_optimizer=None):

    fc1 = AverageMeter('acc1', ':6.2f')
    fc4 = AverageMeter('accz1', ':6.2f')
    fc5 = AverageMeter('accz2', ':6.2f')
    losses1 = AverageMeter('Loss1', ':.4e')
    losses2 = AverageMeter('Loss2', ':.4e')
    losses3 = AverageMeter('Loss3', ':.4e')
    losses4 = AverageMeter('Loss4', ':.4e')
    losses5 = AverageMeter('Loss5', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    # criterion_down = nn.CrossEntropyLoss(weight=torch.FloatTensor([3,3,1])).cuda(args.gpu)
    criterion_down = nn.CrossEntropyLoss().cuda(args.gpu)


    progress = ProgressMeter(
        len(train_loader),
        [losses, fc1, fc4, fc5,  losses1, losses2, losses3, losses4, losses5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (input_ids, attention_masks, segment_ids, target, shadow_target) in enumerate(train_loader):


        input_ids = input_ids.cuda(args.gpu, non_blocking=True)
        attention_masks = attention_masks.cuda(args.gpu, non_blocking=True)
        segment_ids = segment_ids.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        shadow_target = shadow_target.cuda(args.gpu, non_blocking=True)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'token_type_ids': segment_ids
        }
        loss1 = loss2 = loss3 = loss4 = loss5 = None
        output_fx, output_z1_fc, output_z2_fc, z1, output_z2_detached, nodetach_z2, pd4mz1, pd4mz2 = model(**batch)



        loss1 = criterion(output_fx, target)

        shadow_target = shadow_target.view(-1, 1)
        loss2 = torch.sum((shadow_target-output_z1_fc)**2)

        # adjust theta
        threshold_loss3 = F.mse_loss(z1, output_z2_detached, reduction='mean')
        if threshold_loss3 < 0.01:
            loss3 = args.theta*F.mse_loss(z1, output_z2_detached, reduction='mean')
        else:
            loss3 = F.mse_loss(z1, output_z2_detached, reduction='mean')+args.theta*F.mse_loss(z1, output_z2_fc, reduction='mean')

        loss3_bonus = -args.theta*F.mse_loss(z1, output_z2_fc, reduction='mean')
        if threshold_loss3 > 1:
            loss3 = loss3/loss3.detach()
            loss3_bonus = loss3_bonus/loss3_bonus.detach()

        l3d = threshold_loss3.detach()
        l2d = loss2.detach()


        if loss2 > 1:
            loss2 = loss2/loss2.detach()

        loss4 = criterion_down(pd4mz1, target)
        loss5 = criterion_down(pd4mz2, target)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss3_bonus
        # loss = loss1



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        acc1 = accuracy(output_fx, target, args=args, topk=(1, ))
        acc4 = accuracy(pd4mz1, target, args=args, topk=(1,))
        acc5 = accuracy(pd4mz2, target, args=args, topk=(1,))

        losses.update(loss.item(), input_ids.size(0))
        fc1.update(acc1[0].item(), input_ids.size(0))
        fc4.update(acc4[0].item(), input_ids.size(0))
        fc5.update(acc5[0].item(), input_ids.size(0))###

        if loss1 != None:
            losses1.update(loss1.item(), input_ids.size(0))
        else:
            losses1.update(0, input_ids.size(0))

        if loss2 != None:
            losses2.update(l2d.item(), input_ids.size(0))
        else:
            losses2.update(0, input_ids.size(0))

        if loss3 != None:
            losses3.update(l3d.item(), input_ids.size(0))
        else:
            losses3.update(0, input_ids.size(0))

        if loss4 != None:
            losses4.update(loss4.item(), input_ids.size(0))
        else:
            losses4.update(0, input_ids.size(0))

        if loss5 != None:
            losses5.update(loss5.item(), input_ids.size(0))
        else:
            losses5.update(0, input_ids.size(0))

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

        if i % (args.print_freq * 60) == 0:
            print('z1: ')
            print(z1)
            print('z2: ')
            print(nodetach_z2)

    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    tensor_writer.add_scalar('ACC@1/train', fc1.avg, epoch)
    tensor_writer.add_scalar('ACC@2/train', fc4.avg, epoch)
    tensor_writer.add_scalar('ACC@3/train', fc5.avg, epoch)
