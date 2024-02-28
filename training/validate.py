import time
import pandas as pd
import torch
import torch.optim
import torch.utils.data

from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
import os


def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None, datasetname=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        losses1 = AverageMeter('Loss1', ':.4e')
        losses2 = AverageMeter('Loss2', ':.4e')
        losses3 = AverageMeter('Loss3', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, losses1, losses2, losses3],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Val: ')

    model.eval()
    print('******************datasetname is {}******************'.format(datasetname))

    log_path = os.path.join(r'./nyslearning/MyResults', args.predout)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    with torch.no_grad():
        end = time.time()
        pred2csv_flag = True

        for i, (input_ids, attention_masks, segment_ids, target, shadow_target) in enumerate(val_loader):

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
            output_fx, output_z1_fc, output_z2_fc, z1_nobert, output_z2_detached, nodetach_z2, pd4mz1, pd4mz2 = model(**batch)

            pred = torch.cat((target.view(-1,1), output_fx), dim=1).cpu().detach()
            
            predz1 = pd4mz1.cpu().detach()
            predz2 = pd4mz2.cpu().detach()

            # pred_x_fx = torch.cat((output_x_minus_fx2mnli, target.view(-1,1)), dim=1).cpu().detach()

            if pred2csv_flag:
                predy = pred
                predz1y = predz1
                predz2y = predz2

                # predy_x_fx = pred_x_fx
                pred2csv_flag = False
            else:
                predy = torch.cat((predy, pred), dim=0)
                predz1y = torch.cat((predz1y, predz1), dim=0)
                predz2y = torch.cat((predz2y, predz2), dim=0)

                # predy_x_fx = torch.cat((predy_x_fx, pred_x_fx), dim=0)

            loss = criterion(output_fx, target)

            acc1 = accuracy(output_fx, target, topk=(1, ), args=args, datasetname=datasetname)
            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0].item(), input_ids.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)



        all_predy = torch.cat((predy, predz1y, predz2y), dim=-1)

        if datasetname in ['QQP_dev', 'QQP_test', 'PAWS']:
            # sample = pd.DataFrame(predy.numpy(), columns=['0','1','label'])
            sample = pd.DataFrame(all_predy.numpy(), columns=['label', 'o1', 'o2', 'z11', 'z12', 'z21', 'z22'])
            sample.to_csv(os.path.join(log_path, datasetname) + str(epoch) + '.csv', sep=',', index = False)
        else:
            sample = pd.DataFrame(all_predy.numpy(), columns=['label', 'o1', 'o2', 'o3', 'z11', 'z12', 'z13', 'z21', 'z22', 'z23'])
            sample.to_csv(os.path.join(log_path, datasetname) + str(epoch) + '.csv', sep=',', index = False)

        # if  datasetname == 'HANS':
        #     sample = pd.DataFrame(predy_x_fx.numpy(), columns=['0','1','2','label'])
        #     sample.to_csv(os.path.join(log_path, datasetname) + str(epoch) + 'predy_x_fx.csv', sep=',', index = False)


        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if test:
            tensor_writer.add_scalar('loss/test', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/test', top1.avg, epoch)
        else:
            tensor_writer.add_scalar('loss/val', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/val', top1.avg, epoch)

    return top1.avg
