import argparse

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net', type = str, default = 'deeplabv3p',
                        choices = ['deeplabv3p', 'wdeeplabv3p'])
    parser.add_argument('--wn', type = str, default = 'none',
                        choices = ['none', 'haar', 'bior2.2', 'bior3.3', 'bior4.4', 'bior5.5', 'db2'])
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet50', 'resnet101', 'vgg16bn'],
                        help='backbone name (default: resnet101)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', '--learning_ratio', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', '--fine_tuning', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()  #在执行这条命令之前，所有命令行参数都给不会生效
    for key, value in args.__dict__.items():
        if not key.startswith('_'):
            print('{} ==> {}'.format(key.rjust(24), value))


if __name__ == '__main__':
    main()