import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

from utilis.datasets import datasets
from utilis.datasets import Collate_function
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from ops.config import parser
from training.train import train
from training.validate import validate
from utilis.saving import save_checkpoint

from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from models.Mymodel import MyAutoModel
from models.MI import CLUB, mi_config


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)




def main():
    args = parser.parse_args()
    set_seed(args.seed)

    print('INFO: --------args----------')
    for k in list(vars(args).keys()):
        print('INFO: %s: %s' % (k, vars(args)[k]))
    print('INFO: --------args----------\n')

    args.log_path = os.path.join(args.log_base, args.dataset, "log.txt")

    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.classes_num, output_hidden_states=True)
    model = MyAutoModel(args.model_name, config=config, args=args, cls_num=args.classes_num)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        pass



    # define loss function (criterion) and optimizer
    if args.dataset == 'MNLI':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.dataset == 'FEVER':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.dataset == 'QQP':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    criterion_train = nn.CrossEntropyLoss().cuda(args.gpu)


    mi_estimator = None
    mi_optimizer = None
    mi_args = mi_config()
    if args.is_mi:
        mi_estimator = eval(mi_args.estimator_name)(mi_args.sample_dim, mi_args.sample_dim, mi_args.hidden_size).cuda(args.gpu)
        mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr = mi_args.lr)


    # Prepare optimizer and schedule (linear warmup and decay)
    if args.bias_correction:
        betas = (0.9, 0.999)
    else:
        betas = (0.0, 0.000)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=betas,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )


    print("Whole args:\n")
    print(args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'MNLI':
        # /root/slabt/dataset/ + MNLI/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev_matched.tsv')
        testdir = os.path.join(args.data, args.dataset, 'dev_mismatched.tsv')

        # /root/slabt/dataset/ + HANS/ +
        test2dir = os.path.join(args.data, args.sub_dataset, 'hansdev.tsv')
        
    elif args.dataset == 'FEVER':
        # /root/slabt/dataset/ + FEVER/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev.tsv')
        testdir = os.path.join(args.data, args.dataset, 'symmv1.tsv')
        test2dir = os.path.join(args.data, args.dataset, 'symmv2.tsv')

    elif args.dataset == 'QQP':
        # /root/slabt/dataset/ + QQP/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev.tsv')
        testdir = os.path.join(args.data, args.dataset, 'test.tsv')
        test2dir = os.path.join(args.data, args.dataset, 'paws.tsv')

    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)

    if args.evaluate:

        val_dataset = datasets(valdir, tokenizer, args.dataset, args)
        test_dataset = datasets(testdir, tokenizer, args.dataset, args)
        test2_dataset = datasets(test2dir, tokenizer, args.sub_dataset, args)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True,
                                                 collate_fn=Collate_function())

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True,
                                                  collate_fn=Collate_function())
        test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True,
                                                   collate_fn=Collate_function())

        if args.dataset == 'MNLI':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='MNLI_match')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='MNLI_mismatch')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='HANS')
        elif args.dataset == 'FEVER':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='FEVER')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='SYMMv1')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='SYMMv2')
        elif args.dataset == 'QQP':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='QQP_dev')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='QQP_test')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='PAWS')

        return


    train_dataset = datasets(traindir, tokenizer, args.dataset, args)
    val_dataset = datasets(valdir, tokenizer, args.dataset, args)
    test_dataset = datasets(testdir, tokenizer, args.dataset, args)
    test2_dataset = datasets(test2dir, tokenizer, args.sub_dataset, args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    print('\n*****train_dataset len is : {}'.format(len(train_dataset)))
    print('\n*****val_dataset len is : {}'.format(len(val_dataset)))
    print('\n*****test_dataset len is : {}'.format(len(test_dataset)))


    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.batch_size
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)


    # begin to train
    best_acc1 = 0
    for epoch in range(args.epochs):

        train(train_loader, model, criterion_train, optimizer, epoch, scheduler, args, tensor_writer, 
                        mi_estimator=mi_estimator, mi_optimizer=mi_optimizer)

        if args.dataset == 'MNLI':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='MNLI_match')
            acc1 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='MNLI_mismatch')
            acc2 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='HANS')
        elif args.dataset == 'FEVER':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='FEVER')
            acc2 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='SYMMv1')
            acc1 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='SYMMv2')
        elif args.dataset == 'QQP':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='QQP_dev')
            acc2 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='QQP_test')
            acc1 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='PAWS')


        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print('Saving...')
        save_checkpoint({
                'epoch': epoch + 1,
                'model_name': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.log_path, epoch)


if __name__ == '__main__':
    main()
