import argparse

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-data', default='./dataset')
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument ('--dataset', type=str, default="MNLI")
parser.add_argument ('--sub_dataset', type=str, default="HANS")
parser.add_argument('--predout', type=str, default='tmp', help='预测输出')
parser.add_argument('--ae_z_size', default=64, type=int)
parser.add_argument('--loss2weight', default=1, type=int)
parser.add_argument('--z_split_first_size', default=4, type=int)
parser.add_argument('--theta', default=1.0, type=float)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--epochs', default=6, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--do_lower_case', type=bool, default=True)
parser.add_argument ('--classes_num', type=int, default=3)
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float)
parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency (default: 10)')
parser.add_argument('--is_detach', action='store_true')
parser.add_argument('--is_mi', action='store_true')

parser.add_argument('--resume', default='', type=str, help='/MNLI/epoch_7_checkpoint.pth.tar')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--log_base', default='./results', type=str, help='path to save logs')


parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
parser.add_argument('--bias_correction', default=True)
parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
